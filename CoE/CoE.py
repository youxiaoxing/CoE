import os
import json
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from decord import VideoReader
from pymongo import MongoClient
from PIL import Image
import base64
import io
import torch
from transformers import AutoProcessor
import requests
from typing import List, Tuple, Dict, Any
from EventGraph import Graph

class Config:
    def __init__(self, config_path: str = "config.json"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
    
    def get(self, *keys):
        result = self.config
        for key in keys:
            result = result[key]
        return result
    
    def get_dataset_config(self, dataset_name: str):
        return self.config["datasets"][dataset_name]
    
    def get_prompt(self, prompt_type: str, prompt_name: str):
        return self.config["prompts"][prompt_type][prompt_name]

class DatabaseManager:
    def __init__(self, config: Config):
        self.config = config
        mongo_config = config.get("mongo")
        self.client = MongoClient(mongo_config["host"], mongo_config["port"])
        self.db = self.client[mongo_config["database"]]
    
    def get_collection_data(self, dataset_name: str):
        dataset_config = self.config.get_dataset_config(dataset_name)
        collection = self.db[dataset_config["collection"]]
        return list(collection.find(dataset_config["query"]))
    
    def graph_search(self, video_id: str, collection_name: str):
        result = self.db[collection_name].find_one({"video_id": video_id})
        return result["graph"] if result else None

class ModelClient:
    def __init__(self, config: Config):
        self.config = config
        model_config = config.get("model")
        self.clients = [
            OpenAI(base_url=url, api_key=model_config["api_key"]) 
            for url in model_config["clients"]
        ]
        self.current_client_idx = model_config["current_client_idx"]
        self.model_name = model_config["model_name"]
        self.max_tokens = model_config["max_tokens"]
        self.temperature = model_config["temperature"]
    
    def get_client(self):
        return self.clients[self.current_client_idx % len(self.clients)]
    
    def chat_completion(self, messages: List[Dict], **kwargs):
        client = self.get_client()
        return client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature)
        )

class VideoProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.processing_config = config.get("processing")
    
    def load_video(self, video_path: str, is_numpy: bool = False):
        if is_numpy:
            return np.load(video_path)
        else:
            vr = VideoReader(video_path)
            original_fps = vr.get_avg_fps()
            sample_interval = int(original_fps)
            
            frames = []
            for i in range(0, len(vr), sample_interval):
                frame = vr[i]
                frames.append(frame.asnumpy())
            
            return np.stack(frames, axis=0)
    
    def get_video_frames(self, video_inputs, control_max_frames: bool = None, max_num_frames: int = None):
        if control_max_frames is None:
            control_max_frames = self.processing_config["control_max_frames"]
        if max_num_frames is None:
            max_num_frames = self.processing_config["max_num_frames"]
            
        if control_max_frames and video_inputs is not None:
            total_frames = video_inputs.shape[0]
            indices = np.linspace(0, total_frames - 1, max_num_frames, dtype=int)
            if total_frames - 1 not in indices:
                indices = np.append(indices, total_frames - 1)
            video_inputs = video_inputs[indices]
        
        return video_inputs

class EventAnalyzer:
    def __init__(self, config: Config, model_client: ModelClient, video_processor: VideoProcessor):
        self.config = config
        self.model_client = model_client
        self.video_processor = video_processor
        self.processing_config = config.get("processing")
        
    def match_subevent(self, video_inputs, subevent_list, prompt_type: str = "vista"):
        total_frames = len(video_inputs)
        frames_per_group = self.processing_config["frames_per_group"]
        frame_groups = (total_frames + frames_per_group - 1) // frames_per_group
        
        def process_group(group_idx):
            start_idx = group_idx * frames_per_group
            end_idx = min(start_idx + frames_per_group, total_frames)
            frames = video_inputs[start_idx:end_idx]
            
            group_content = []
            for frame in frames:
                pil_image = Image.fromarray(frame)
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                group_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_str}"}
                })
            
            prompt_template = self.config.get_prompt(prompt_type, "subevent_match")
            subevent_text = chr(10).join(f"{i} {e}" for i, e in enumerate(subevent_list))
            prompt = prompt_template.format(subevent_list=subevent_text)
            
            group_content.append({"type": "text", "text": prompt})
            
            messages = [{"role": "system", "content": "You are a precise visual event annotator. Using only the provided frames and sub-event list, pick the best-matching sub-event for each frame group."},
                        {"role": "user", "content": group_content}]
            response = self.model_client.chat_completion(messages)
            
            content = response.choices[0].message.content
            try:
                content = json.loads(content.replace("```json", "").replace("```", "").replace("\n", ""))
            except Exception:
                return None
            
            if isinstance(content, list):
                content = content[0]
            
            return {
                "group_idx": group_idx,
                "frames": list(range(start_idx + 1, end_idx + 1)),
                "description": content
            }
        
        results = []
        max_workers = self.processing_config["max_workers"]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(process_group, i) for i in range(frame_groups)]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        
        results.sort(key=lambda x: x["group_idx"])
        return [{"frames": r["frames"], "description": r["description"]} for r in results]
    
    def match_video_entities(self, video_inputs, subgraph, prompt_type: str = "vista"):
        content = []
        for frame_idx in range(video_inputs.shape[0]):
            frame = video_inputs[frame_idx]
            pil_image = Image.fromarray(frame)
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_str}"}
            })
        
        prompt_template = self.config.get_prompt(prompt_type, "entity_match")
        prompt = prompt_template.format(subgraph=subgraph)
        content.append({"type": "text", "text": prompt})
        
        messages = [
            {"role": "system", "content": "You are a cautious video entity & relation matcher."},
            {"role": "user", "content": content}
        ]
        
        response = self.model_client.chat_completion(messages)
        
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            return content.replace("```json", "").replace("```", "").replace("\n", "")
        
        return None
    
    def get_summary(self, describe_list, total_event, subevent_list, storyline, entity_dict=None, entity_relation=None, prompt_type: str = "vista"):
        prompt_template = self.config.get_prompt(prompt_type, "summary_generation")
        prompt = prompt_template.format(
            total_event=total_event,
            sub_events=", ".join(subevent_list),
            entity_dict=entity_dict,
            entity_relation=entity_relation,
            scene_descriptions=json.dumps(describe_list),
            storyline=storyline
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that writes precise, factual summaries in English."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.model_client.chat_completion(messages)
        
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            return content.replace("```json", "").replace("```", "").replace("\n", "")
        
        return None

class QuestEvaluator:
    def __init__(self, config: Config, model_client: ModelClient):
        self.config = config
        self.model_client = model_client
        self.bert_score_url = config.get("bert_score", "server_url")
        self.f1_threshold = config.get("processing", "f1_threshold")
    
    def cal_score(self, ref_list: List[str], answer_list: List[str]) -> Tuple[float, List[float]]:
        try:
            response = requests.post(
                self.bert_score_url,
                json={'refList': ref_list, 'answerList': answer_list},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result['average_score'], result['bert_scores']
        except requests.exceptions.RequestException as e:
            print(f"Error calling BERTScore service: {e}")
            return 0.0, [0.0] * len(ref_list)
    
    def quest_eval(self, sub_events, graph_list, summary):
        need_refine_list = []
        redundancy_list = []
        ignore_list = []
        f1 = 0
        
        for index, graph in enumerate(graph_list):
            if graph.isEmpty():
                ignore_list.append([])
                redundancy_list.append([])
                continue
            
            graph_que_aner_list = self._generate_graph_question_answer(
                sub_events[index] + graph.get_graph_string()
            )
            summary_que_aner_list = self._generate_summary_question_answer(summary)
            
            if not graph_que_aner_list or not summary_que_aner_list:
                ignore_list.append([])
                redundancy_list.append([])
                continue
            
            graph_que_list = [item["question"] for item in graph_que_aner_list]
            graph_aner_list = [item["answer"] for item in graph_que_aner_list]
            summary_que_list = [item["question"] for item in summary_que_aner_list]
            summary_aner_list = [item["answer"] for item in summary_que_aner_list]
            
            graph_gen_aner_list = self._answer_graph_question(graph_que_list, summary)
            summary_gen_aner_list = self._answer_summary_question(
                summary_que_list, sub_events[index] + graph.get_graph_string()
            )
            
            recall, recall_scores = self.cal_score(graph_aner_list, graph_gen_aner_list)
            precision, precision_scores = self.cal_score(summary_aner_list, summary_gen_aner_list)
            
            if recall > 0 and precision > 0:
                f1 = 2 * (recall * precision) / (precision + recall)
            else:
                f1 = 0
            
            if f1 < self.f1_threshold:
                need_refine_list.append(index)
                ignore = []
                redundancy = []
                
                for idx, (recall_score, precision_score) in enumerate(zip(recall_scores, precision_scores)):
                    if recall_score < recall:
                        ignore.append(graph_que_aner_list[idx])
                    if precision_score < precision:
                        redundancy.append(summary_que_aner_list[idx])
                
                ignore_list.append(ignore)
                redundancy_list.append(redundancy)
            else:
                ignore_list.append([])
                redundancy_list.append([])
        
        return need_refine_list, redundancy_list, ignore_list, f1
    
    def _generate_graph_question_answer(self, subgraph):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f'''Based on the given subgraph, please create questions and answers about the important entities and relationships in the subgraph. The answers should be concise.
             Return in JSON format as follows:
             [{{"question": "question 1", "answer": "answer 1"}}, {{"question": "question 2", "answer": "answer 2"}}]
             
             Note that the number of questions should not be too many, only focus on the most important entities or relationships.
             
             Here is the event subgraph:
             {subgraph}'''}
        ]
        
        response = self.model_client.chat_completion(messages)
        
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            try:
                return json.loads(content.replace("```json", "").replace("```", "").replace("\n", ""))
            except:
                return None
        return None
    
    def _generate_summary_question_answer(self, summary):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f'''Based on the given summary, please generate questions and answers about the most important entities, perspectives, etc. Make sure the answers are concise and brief.
             Return in JSON format as follows:
             [{{"question": "question 1", "answer": "answer 1"}}, {{"question": "question 2", "answer": "answer 2"}}]
             
             Note that the number of questions should not be too many, only focus on the most important entities or viewpoints.
            
             Here is the summary:
             {summary}'''}
        ]
        
        response = self.model_client.chat_completion(messages)
        
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            try:
                return json.loads(content.replace("```json", "").replace("```", "").replace("\n", ""))
            except:
                return None
        return None
    
    def _answer_graph_question(self, question_list, summary):
        answer_list = []
        for question in question_list:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f'''Please answer this question based on the given summary. Note that the answer should be sufficiently concise.
                
                Here is the question:
                {question}
                Here is the summary:
                {summary}'''}
            ]
            
            response = self.model_client.chat_completion(messages)
            
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                content = content.replace("```json", "").replace("```", "").replace("\n", "")
                answer_list.append(content)
        
        return answer_list
    
    def _answer_summary_question(self, question_list, subgraph):
        answer_list = []
        for question in question_list:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f'''Please answer this question based on the event subgraph. Note that the answer should be sufficiently concise.
                
                Here is the question:
                {question}
                Here is the subgraph:
                {subgraph}'''}
            ]
            
            response = self.model_client.chat_completion(messages)
            
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                content = content.replace("```json", "").replace("```", "").replace("\n", "")
                answer_list.append(content)
        
        return answer_list

class VideoEventProcessor:
    def __init__(self, config_path: str = "config.json"):
        self.config = Config(config_path)
        os.environ["HF_ENDPOINT"] = self.config.get("hf_endpoint")

        self.db_manager = DatabaseManager(self.config)
        self.model_client = ModelClient(self.config)
        self.video_processor = VideoProcessor(self.config)
        self.event_analyzer = EventAnalyzer(self.config, self.model_client, self.video_processor)
        self.quest_evaluator = QuestEvaluator(self.config, self.model_client)
    
    def process_dataset(self, dataset_name: str):
        dataset_config = self.config.get_dataset_config(dataset_name)
        results = self.db_manager.get_collection_data(dataset_name)
        results.reverse()
        def process_video(item):
            return self._process_single_video(item, dataset_config)
        
        max_workers = self.config.get("processing", "max_workers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(process_video, results), total=len(results)))
    
    def _process_single_video(self, item, dataset_config):
        video_id = item["video_id"]
        item["_id"] = str(item["_id"])
        storyline = item[dataset_config["article_field"]]
        storyline = " ".join(storyline.split()[:4096])
        if self._is_already_processed(item, dataset_config["save_file"]):
            return
        
        try:
            video_path = dataset_config["video_path_template"].format(**item)
            is_numpy = video_path.endswith('.npy')
            video_inputs = self.video_processor.load_video(video_path, is_numpy)
            video_inputs = self.video_processor.get_video_frames(video_inputs)
            
            graph = self.db_manager.graph_search(video_id, dataset_config["collection"])
            if not graph:
                print(f"[{video_id}] No graph found")
                return
            
            total_event = graph["total_event"]
            subevent_list = graph["sub_event"]
            graph_list = [Graph() for _ in range(len(subevent_list))]
            
            frame_descriptions = self.event_analyzer.match_subevent(
                video_inputs, subevent_list, dataset_config["prompt_type"]
            )

            relations_list = self._process_entity_relations(
                video_inputs, frame_descriptions, graph, dataset_config["prompt_type"]
            )
            
            describe_list = self._merge_frame_descriptions_by_graph(
                relations_list, graph_list, video_inputs, frame_descriptions, storyline
            )
            
            final_summary = self._generate_and_evaluate_summary(
                describe_list, total_event, subevent_list, graph, graph_list, 
                relations_list, video_inputs, frame_descriptions, storyline, dataset_config["prompt_type"], need_evaluate=False
            )

            final_summary = self.translate_style(final_summary, dataset_config["prompt_type"])

            self._save_result(item, final_summary, dataset_config["save_file"])
            
        except Exception as e:
            print(f"[{video_id}] Error: {e}")
    
    def translate_style(self, final_summary, prompt_type):
        prompt_template = self.config.get_prompt(prompt_type, "translate_style")
        prompt = prompt_template.format(
            final_summary=final_summary
        )

        messages = [
            {"role": "system", "content": "You are a precise style editor. Rewrite the given summary into the requested style and tone while preserving all facts and meaning."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.model_client.chat_completion(messages)
        
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            return content

    def _is_already_processed(self, item, save_file):
        if not os.path.exists(save_file):
            return False
        
        with open(save_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if '_id' in data and data['_id'] == str(item["_id"]):
                        return True
                except json.JSONDecodeError:
                    continue
        return False
    
    def _save_result(self, item, summary, save_file):
        save_item = {
            "_id": str(item["_id"]),
            "response": summary
        }
        
        with open(save_file, "a") as f:
            json.dump(save_item, f, ensure_ascii=True)
            f.write("\n")
            f.flush()
    
    def _process_entity_relations(self, video_inputs, frame_descriptions, graph, prompt_type):
        relations_list = []
        
        def process_segment(idx, frame_description):
            start_idx = frame_description["frames"][0] - 1
            end_idx = frame_description["frames"][-1]
            description = frame_description["description"]
            subevent_idx = int(description["idx"])
            
            if subevent_idx >= len(graph["entity_dict"]):
                subevent_idx = len(graph["entity_dict"]) - 1

            if subevent_idx == -1:
                return []
            sub_entities = graph["entity_dict"][subevent_idx]
            sub_entity_relations = graph["entity_relation"][subevent_idx]
            
            subgraph = {
                "sub event": description["subevent"],
                "entities": sub_entities,
                "entities relations": sub_entity_relations
            }
            
            relations = self.event_analyzer.match_video_entities(
                video_inputs[start_idx:end_idx], json.dumps(subgraph), prompt_type
            )
            
            return relations
        
        max_workers = self.config.get("processing", "max_workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_segment, i, fd): i 
                    for i, fd in enumerate(frame_descriptions)}
            relations_list = [None] * len(frame_descriptions)
            for fut in as_completed(futures):
                idx = futures[fut]
                relations_list[idx] = fut.result()
        
        return relations_list
    
    def _merge_frame_descriptions_by_graph(self, relations_list, graph_list, video_inputs, frame_descriptions, storyline):
        max_merge = self.config.get("processing", "max_segments")
        describe_list = []
        
        buffer_clips = []
        buffer_frames = []
        buffer_new_relations = []
        buffer_subevents = []
        last_graph_idx = None
        
        def flush_buffer():
            if not buffer_frames:
                return
            start = buffer_frames[0][0] - 1
            end = buffer_frames[-1][-1]
            clip = np.concatenate(buffer_clips, axis=0)
            subevents = [s for s in buffer_subevents if s]
            merged_subevent = "; ".join(subevents)
            new_relation = sum(buffer_new_relations, [])
            
            discribe = self._extract_video_split(
                clip, merged_subevent,
                graph_list[last_graph_idx].get_graph_string(),
                new_relation,
                storyline
            )
            describe_list.append({
                "time": f"{start + 1}: {end}s",
                "description": discribe
            })
        
        for relations, description in zip(relations_list, frame_descriptions):
            current_desc = description["description"]
            current_idx = int(current_desc["idx"])
            new_relation = []
            
            if relations:
                try:
                    if "triples" not in relations:
                        relations = json.loads(relations)
                    else:
                        relations = json.loads(relations)["triples"]
                    for relation in relations:
                        if (graph_list[current_idx].add_node(relation["from"]) or
                            graph_list[current_idx].add_node(relation["to"]) or
                            graph_list[current_idx].add_edge(
                                relation["from"], relation["to"], relation["relation"]
                            )):
                            new_relation.append(relation)
                except:
                    pass
            
            frames = description["frames"]
            start = frames[0] - 1
            end = frames[-1]
            clip = video_inputs[start:end]
            
            if new_relation or len(buffer_clips) >= max_merge or current_idx != last_graph_idx:
                flush_buffer()
                buffer_clips = [clip]
                buffer_frames = [frames]
                buffer_new_relations = [new_relation]
                buffer_subevents = [current_desc["subevent"]]
            else:
                buffer_clips.append(clip)
                buffer_frames.append(frames)
                buffer_new_relations.append([])
                buffer_subevents.append(current_desc["subevent"])
            
            last_graph_idx = current_idx
        
        flush_buffer()
        return describe_list
    
    def _describe_video_split(self, video_inputs, sub_event, subgraph, new_relation):
        content = []
        
        for frame_idx in range(video_inputs.shape[0]):
            frame = video_inputs[frame_idx]
            pil_image = Image.fromarray(frame)
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_str}"}
            })
        
        content.append({
            "type": "text",
            "text": f'''Based on the video clip, the provided event and event subgraph, and the new entities and relationships that appeared from the previous clip to the current one, please analyze the event trajectory and generate a brief description text not exceeding 100 words.
            Output format:
            Event trajectory: analyzing text.
            Description: text.

            Here is the input:
            sub event: {sub_event}
            sub graph: {subgraph}
            new entities and relations: {new_relation}'''
        })
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ]
        
        response = self.model_client.chat_completion(messages)
        
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            return content.replace("```json", "").replace("```", "").replace("\n", "")
        
        return None
    

    def _extract_video_split(self, video_inputs, sub_event, subgraph, new_relation, storyline):
        content = []
        
        for frame_idx in range(video_inputs.shape[0]):
            frame = video_inputs[frame_idx]
            pil_image = Image.fromarray(frame)
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_str}"}
            })
        
        content.append({
            "type": "text",
            "text": f'''
            Based on the reference transcript, the given event and sub-event graph, and the newly identified entities and relations, extract the most relevant information from the transcript.  
Then analyze the event trajectory and generate a concise description in no more than 100 words.  
STRICT OUTPUT FORMAT (no extra text):  
Event trajectory: <analysis of event progression>  
Description summary: <concise summary>  

Inputs:  
- Reference transcript: {storyline}  
- Sub-event: {sub_event}  
- Subgraph: {subgraph}  
- New entities and relations: {new_relation}
'''
        })
        
        messages = [
            {"role": "system", "content": "You are a grounded video-clip explainer. Rely only on the provided frames, transcript, subgraph, and new relations; do not invent facts."},
            {"role": "user", "content": content}
        ]
        
        response = self.model_client.chat_completion(messages)
        
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            return content.replace("```json", "").replace("```", "").replace("\n", "")
        
        return None
    
    def _generate_and_evaluate_summary(self, describe_list, total_event, subevent_list, 
                                     graph, graph_list, relations_list, video_inputs, 
                                     frame_descriptions, storyline, prompt_type, need_evaluate=True):
        summary = self.event_analyzer.get_summary(
            describe_list, total_event, subevent_list, storyline,
            graph["entity_dict"], graph["entity_relation"], prompt_type
        )

        original_f1 = 0
        final_summary = summary
        
        if need_evaluate:
            need_refine_list, redundancy_list, ignore_list, f1 = self.quest_evaluator.quest_eval(
                subevent_list, graph_list, summary
            )
            
            if f1 >= original_f1:
                original_f1 = f1
                final_summary = summary

            max_iterations = self.config.get("processing", "quest_eval_iterations")
            count = 0
            
            while count < max_iterations and len(need_refine_list) > 0:
                describe_list = self._refine(
                    relations_list, graph_list, video_inputs, frame_descriptions,
                    need_refine_list, describe_list, redundancy_list, ignore_list, storyline
                )
                
                summary = self.event_analyzer.get_summary(
                    describe_list, total_event, subevent_list, storyline,
                    graph["entity_dict"], graph["entity_relation"], prompt_type
                )
                
                need_refine_list, redundancy_list, ignore_list, f1 = self.quest_evaluator.quest_eval(
                    subevent_list, graph_list, summary
                )
                
                count += 1
                if f1 > original_f1:
                    original_f1 = f1
                    final_summary = summary
        
        return final_summary
    
    def _refine(self, relations_list, graph_list, video_inputs, frame_descriptions,
               need_refine_list, describe_list, redundancy_list, ignore_list, storyline):
        for idx in need_refine_list:
            graph_list[idx].empty()
        
        new_describe_list = describe_list.copy()
        max_merge = self.config.get("processing", "max_segments")
        
        buffer_clips = []
        buffer_frames = []
        buffer_new_relations = []
        buffer_subevents = []
        last_graph_idx = None
        current_describe_idx = 0
        
        def flush_buffer():
            nonlocal current_describe_idx
            if not buffer_frames:
                return
            
            start = buffer_frames[0][0] - 1
            end = buffer_frames[-1][-1]
            clip = np.concatenate(buffer_clips, axis=0)
            subevents = [s for s in buffer_subevents if s]
            merged_subevent = "; ".join(subevents)
            new_relation = sum(buffer_new_relations, [])
            
            if last_graph_idx in need_refine_list:
                discribe = self._extract_video_split_refine(
                    clip, merged_subevent,
                    graph_list[last_graph_idx].get_graph_string(),
                    new_relation,
                    redundancy_list[last_graph_idx], 
                    ignore_list[last_graph_idx],
                    storyline
                )
                
                new_description = {
                    "time": f"{start + 1}: {end}s",
                    "description": discribe
                }
                
                if current_describe_idx < len(new_describe_list):
                    new_describe_list[current_describe_idx] = new_description
                else:
                    new_describe_list.append(new_description)
            
            current_describe_idx += 1
        
        flag = True
        for relations, description in zip(relations_list, frame_descriptions):
            current_desc = description["description"]
            current_idx = int(current_desc["idx"])
            
            if flag:
                last_graph_idx = current_idx
                flag = False
            
            should_process = current_idx in need_refine_list
            new_relation = []
            
            if should_process and relations:
                try:
                    if "triples" not in relations:
                        relations = json.loads(relations)
                    else:
                        relations = json.loads(relations)["triples"]
                    for relation in relations:
                        if (graph_list[current_idx].add_node(relation["from"]) or
                            graph_list[current_idx].add_node(relation["to"]) or
                            graph_list[current_idx].add_edge(
                                relation["from"], relation["to"], relation["relation"]
                            )):
                            new_relation.append(relation)
                except:
                    pass
            
            frames = description["frames"]
            start = frames[0] - 1
            end = frames[-1]
            clip = video_inputs[start:end]
            
            if new_relation or len(buffer_clips) >= max_merge or current_idx != last_graph_idx:
                flush_buffer()
                buffer_clips = [clip]
                buffer_frames = [frames]
                buffer_new_relations = [new_relation]
                buffer_subevents = [current_desc["subevent"]]
            else:
                buffer_clips.append(clip)
                buffer_frames.append(frames)
                buffer_new_relations.append([])
                buffer_subevents.append(current_desc["subevent"])
            
            last_graph_idx = current_idx
        
        flush_buffer()
        return new_describe_list
    
    def _describe_video_split_refine(self, video_inputs, sub_event, subgraph, 
                                   new_relation, redundancy_list, ignore_list):
        content = []
        
        for frame_idx in range(video_inputs.shape[0]):
            frame = video_inputs[frame_idx]
            pil_image = Image.fromarray(frame)
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_str}"}
            })
        
        content.append({
            "type": "text",
            "text": f'''Based on the video clip, the provided event and event subgraph, and the new entities and relationships that appeared from the previous clip to the current one, please analyze the event trajectory and generate a brief description text not exceeding 100 words.
            In addition, there will some information you need to ignore or focus. 
            Output format:
            Event trajectory: analyzing text.
            Summary: text.

            Here is the input:
            Focus questions: {"".join([json.dumps(item) for item in ignore_list])}
            Ignore questions: {"".join([json.dumps(item) for item in redundancy_list])}
            sub event: {sub_event}
            sub graph: {subgraph}
            new entities and relations: {new_relation}'''
        })
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ]
        
        response = self.model_client.chat_completion(messages)
        
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            return content.replace("```json", "").replace("```", "").replace("\n", "")
        
        return None
    
    def _extract_video_split_refine(self, video_inputs, sub_event, subgraph, 
                                   new_relation, redundancy_list, ignore_list, storyline):
        content = []
        
        for frame_idx in range(video_inputs.shape[0]):
            frame = video_inputs[frame_idx]
            pil_image = Image.fromarray(frame)
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_str}"}
            })
        
        content.append({
            "type": "text",
            "text": f'''
            Based on the reference transcript, the given event, the sub-event graph, and the newly identified entities and relations, extract the most relevant information from the transcript.  
Analyze the event trajectory and generate a concise description in no more than 100 words.
IMPORTANT:  
- Focus especially on the aspects listed under "Focus questions".  
- Ignore or downplay content that matches "Ignore questions".  

STRICT OUTPUT FORMAT (no extra text):  
Event trajectory: <analysis of event progression>  
Summary: <concise description>  

Inputs:  
- Focus questions: {"".join([json.dumps(item) for item in ignore_list])}  
- Ignore questions: {"".join([json.dumps(item) for item in redundancy_list])}  
- Reference transcript: {storyline}  
- Sub-event: {sub_event}  
- Subgraph: {subgraph}  
- New entities and relations: {new_relation}'''
        })
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ]
        
        response = self.model_client.chat_completion(messages)
        
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            return content.replace("```json", "").replace("```", "").replace("\n", "")
        
        return None

import argparse

def main():
    parser = argparse.ArgumentParser(description="Video Event Processing")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)

    args = parser.parse_args()

    processor = VideoEventProcessor(args.config)
    processor.process_dataset(args.dataset)

if __name__ == "__main__":
    main()