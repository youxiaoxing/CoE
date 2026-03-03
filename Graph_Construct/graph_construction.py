import json
import os
from typing import Dict, List, Optional, Any
from tqdm import tqdm
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI


class Config:
    
    def __init__(self, config_path: str = "config.json"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
    
    def get_mongo_config(self) -> Dict[str, Any]:
        return self.config.get("mongo", {})
    
    def get_model_config(self) -> Dict[str, Any]:
        return self.config.get("model", {})
    
    def get_processing_config(self) -> Dict[str, Any]:
        return self.config.get("processing", {})
    
    def get_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        return self.config.get("datasets", {}).get(dataset_name, {})
    
    def get_prompts_config(self, prompt_type: str) -> Dict[str, str]:
        return self.config.get("prompts", {}).get(prompt_type, {})


class EventGraphExtractor:
    
    def __init__(self, config: Config):
        self.config = config
        self.model_config = config.get_model_config()
        self.clients = self._init_clients()
        self.current_client_idx = self.model_config.get("current_client_idx", 0)
    
    def _init_clients(self) -> List[OpenAI]:
        clients = []
        api_key = self.model_config.get("api_key", "-")
        
        for base_url in self.model_config.get("clients", []):
            client = OpenAI(
                base_url=base_url,
                api_key=os.getenv("VLLM_API_KEY", api_key)
            )
            clients.append(client)
        
        return clients
    
    def _get_client(self) -> OpenAI:
        if not self.clients:
            raise ValueError("No OpenAI clients configured")
        return self.clients[self.current_client_idx % len(self.clients)]
    
    def _make_api_call(self, prompt: str, system_message: str = "You are a helpful assistant.") -> Optional[str]:
        try:
            client = self._get_client()
            model_name = self.model_config.get("model_name", "Qwen2.5-72B-Instruct-AWQ")
            
            chat_completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.model_config.get("max_tokens", 2000),
                temperature=self.model_config.get("temperature", 0.1)
            )
            
            if chat_completion.choices and len(chat_completion.choices) > 0:
                content = chat_completion.choices[0].message.content
                return content.replace("```json", "").replace("```", "").replace("\n", "")
            
            return None
        except Exception as e:
            print(f"API call failed: {e}")
            self.current_client_idx = (self.current_client_idx + 1) % len(self.clients)
            return None
    
    def get_total_event(self, text: str) -> Optional[str]:
        prompt = self._get_total_event_prompt(text)
        return self._make_api_call(prompt)
    
    def _get_total_event_prompt(self, text: str) -> str:
        return f'''Summarize this news event in one sentence, ensuring your summary captures the high-level essence that represents the entire news story.
Keep your summary concise, preferably under 25 words.
Here are two examples:
1. Egyptian Foreign Minister offered an 'open invitation' for reconciliation among all groups while urging an end to violencey.
2. A suicide car bombing occurred near Kabul Airport in Afghanistan.

The following is the news article:
{text}
'''
    
    def get_subevent(self, summary: str, text: str) -> Optional[str]:
        prompt = self._get_subevent_prompt(summary, text)
        return self._make_api_call(prompt)
    
    def _get_subevent_prompt(self, summary: str, text: str) -> str:
        return f'''
{summary}

Above is the main event summary of the news article below. Please analyze if this news article can be further divided into sub-events (maximum 3).

Return your response in JSON format:
- If the article CANNOT be meaningfully divided into sub-events, return:
  {{"result": "False", "events": []}}
  
- If the article CAN be divided into sub-events, return:
  {{"result": "True", "events": ["event1", "event2", ...]}}
  
Each sub-event should be a concise one-sentence summary, following the same format as the main summary. Keep the granularity appropriate - don't make the sub-events too specific or detailed.

The following is the news article:
{text}
'''
    
    def get_entities(self, event: str, text: str) -> Optional[str]:
        prompt = self._get_entities_prompt(event, text)
        return self._make_api_call(prompt)
    
    def _get_entities_prompt(self, event: str, text: str) -> str:
        return f'''
Please extract relevant entities related to the specified event from the provided news text. Entities should include people, locations, organizations, and objects/items. Return the results in JSON format with the following structure:

{{
"person": ["name1", "name2", "name3"],
"location": ["place1", "place2", "place3"],
"organization": ["org1", "org2", "org3"],
"item": ["item1", "item2", "item3"]
}}

Instructions:
1. Only include entities directly related to the specified event
2. Remove duplicates and normalize entity names
3. For people, use full names when available
4. For organizations, use official names rather than abbreviations when possible
5. **Ensure all extracted entities are actually mentioned in the text**
6. If no entities of a certain type are found, return an empty array for that category

Event: {event}
News text: {text}
'''
    
    def get_entities_relation(self, entities_text: str, text: str) -> Optional[str]:
        prompt = self._get_entities_relation_prompt(entities_text, text)
        return self._make_api_call(prompt)
    
    def _get_entities_relation_prompt(self, entities: str, text: str) -> str:
        return f'''
Please identify the relationships between entities in the news article. Strictly adhere to the information provided in the text. Return the results as a JSON list, where each item contains (entity a, relationship, entity b).

Format: {[{"from": "Name A", "relationship": "specific relationship", "to": "Name B"}, {"from": "Name A", "relationship": "specific relationship", "to": "Name C"}]}

When analyzing, please:
1. Only include relationships explicitly stated in the text
2. Only include entities that appear in the provided entities list
3. Use precise relationship descriptions (e.g., "is CEO of" rather than just "works at")
4. JSON format uses double quotation marks

Entities list: {entities}
News text: {text}
'''


class DatasetProcessor:
    def __init__(self, config: Config, dataset_name: str):
        self.config = config
        self.dataset_name = dataset_name
        self.dataset_config = config.get_dataset_config(dataset_name)
        self.processing_config = config.get_processing_config()
        self.extractor = EventGraphExtractor(config)
        self.db = self._init_database()
    
    def _init_database(self) -> Any:
        mongo_config = self.config.get_mongo_config()
        client = MongoClient(
            mongo_config.get("host", "localhost"),
            mongo_config.get("port", 27017)
        )
        return client[mongo_config.get("database", "views")]
    
    def get_article_text(self, item: Dict[str, Any]) -> str:
        article_field = self.dataset_config.get("article_field", "storyline")
        return item.get(article_field, "")
    
    def process_item(self, item: Dict[str, Any]) -> bool:
        try:
            # item["_id"] = str(item["_id"])
            save_json = {}
            storyline = self.get_article_text(item)
            
            if not storyline:
                return False
            

            total_event = self.extractor.get_total_event(storyline)
            if not total_event:
                return False
            

            sub_event_response = self.extractor.get_subevent(total_event, storyline)
            if not sub_event_response:
                return False
            
            try:
                sub_event = json.loads(sub_event_response)
            except json.JSONDecodeError:
                return False
            
            save_json["total_event"] = total_event
            

            if sub_event.get("result") == "False":
                sub_event["events"] = [total_event]
            
            save_json["sub_event"] = sub_event["events"]
            

            entity_dict_list = []
            relation_dict_list = []
            
            for subevent in sub_event["events"]:

                entity_response = self.extractor.get_entities(subevent, storyline)
                if not entity_response:
                    continue
                
                try:
                    entity_dict = json.loads(entity_response)
                except json.JSONDecodeError:
                    continue
                

                entity_list = set()
                entity_text = ""
                for key in entity_dict.keys():
                    for entity in entity_dict[key]:
                        entity_text += (entity + ", ")
                        entity_list.add(entity)
                

                entities_relation_response = self.extractor.get_entities_relation(entity_text, storyline)
                if entities_relation_response:
                    try:
                        entities_relation = json.loads(entities_relation_response)

                        filtered_entities_relation = []
                        entities_relation = [rel for rel in entities_relation if "to" in rel]
                        
                        for relation in entities_relation:
                            if relation["from"] in entity_list and relation["to"] in entity_list:
                                filtered_entities_relation.append(relation)
                        
                        entities_relation = filtered_entities_relation
                    except json.JSONDecodeError:
                        entities_relation = []
                else:
                    entities_relation = []
                
                entity_dict_list.append(entity_dict)
                relation_dict_list.append(entities_relation)
            
            save_json["entity_dict"] = entity_dict_list
            save_json["entity_relation"] = relation_dict_list

            collection_name = self.dataset_config.get("collection", self.dataset_name)
            collection = getattr(self.db, collection_name)
            result = collection.update_one(
                {"_id": item["_id"]}, 
                {"$set": {"graph": save_json}}
            )


            if result.matched_count == 0:
                print(f"Item {item.get('_id', 'unknown')}: No document found with this _id")
                return False
            elif result.modified_count == 0:
                print(f"Item {item.get('_id', 'unknown')}: Document found but no changes made (data might be identical)")
                return True 
            else:
                print(f"Item {item.get('_id', 'unknown')}: Successfully updated in database")
                return True
            
            return True
            
        except Exception as e:
            print(f"Error processing item {item.get('_id', 'unknown')}: {e}")
            return False
    
    def process_dataset(self) -> None:
        collection_name = self.dataset_config.get("collection", self.dataset_name)
        collection = getattr(self.db, collection_name)
        
        base_query = self.dataset_config.get("query", {})
        query = {
            **base_query,
            "graph": {"$exists": False}
        }

        results = list(collection.find(query))
        results.reverse()
        
        if not results:
            print(f"No items to process for dataset {self.dataset_name}")
            return
        
        print(f"Processing {len(results)} items for dataset {self.dataset_name}")

        max_workers = self.processing_config.get("max_workers", 1)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_item, item): item for item in results}
            
            success_count = 0
            for future in tqdm(as_completed(futures), total=len(futures)):
                if future.result():
                    success_count += 1
            
            print(f"Successfully processed {success_count}/{len(results)} items")


class EventGraphPipeline:

    def __init__(self, config_path: str = "config.json"):
        self.config = Config(config_path)
    
    def process_dataset(self, dataset_name: str) -> None:
        if dataset_name not in self.config.config.get("datasets", {}):
            raise ValueError(f"Dataset {dataset_name} not found in configuration")
        
        processor = DatasetProcessor(self.config, dataset_name)
        processor.process_dataset()
    
    def process_all_datasets(self) -> None:
        datasets = self.config.config.get("datasets", {})
        for dataset_name in datasets.keys():
            print(f"Processing dataset: {dataset_name}")
            self.process_dataset(dataset_name)
            print(f"Completed dataset: {dataset_name}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Event Graph Extraction Pipeline")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument("--dataset", help="Specific dataset to process (if not provided, processes all datasets)")
    
    args = parser.parse_args()
    
    pipeline = EventGraphPipeline(args.config)
    
    if args.dataset:
        pipeline.process_dataset(args.dataset)
    else:
        pipeline.process_all_datasets()


if __name__ == "__main__":
    main()