import json
import argparse
import os
from pymongo import MongoClient

class DatasetProcessor:
    def __init__(self, config_path="dataset_config.json"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 设置HF-Mirror环境变量
        if 'hf_endpoint' in self.config:
            os.environ['HF_ENDPOINT'] = self.config['hf_endpoint']
            print(f"已设置HF_ENDPOINT为: {self.config['hf_endpoint']}")
    
    def get_mongo_client(self):
        """创建MongoDB客户端连接"""
        mongo_config = self.config['mongo']
        return MongoClient(mongo_config['host'], mongo_config['port'])
    
    def get_mongo_data(self, dataset_config):
        """从MongoDB获取数据"""
        client = self.get_mongo_client()
        
        try:
            db = client[self.config['mongo']['database']]
            collection = db[dataset_config['collection']]
            
            query = dataset_config.get('query', {})
            results = list(collection.find(query))
            print(f"从MongoDB获取到 {len(results)} 条记录")
            
        finally:
            client.close()
        
        return results

    def get_dataset_config(self, dataset_name):
        """获取指定数据集的配置"""
        if dataset_name not in self.config['datasets']:
            available_datasets = list(self.config['datasets'].keys())
            raise ValueError(f"数据集 '{dataset_name}' 不存在于配置中。可用数据集: {available_datasets}")
        
        return self.config['datasets'][dataset_name]

    def change_jsonl(self, dataset_name):
        """处理JSONL文件并生成最终结果"""
        dataset_config = self.get_dataset_config(dataset_name)
        
        print(f"开始处理数据集: {dataset_name}")
        print(f"读取文件: {dataset_config['save_file']}")
        
        # 读取JSONL文件
        try:
            with open(dataset_config['save_file'], 'r', encoding='utf-8') as f:
                result_list = [json.loads(line) for line in f]
            print(f"读取到 {len(result_list)} 条JSONL记录")
        except FileNotFoundError:
            print(f"错误: 文件 {dataset_config['save_file']} 不存在")
            return
        except json.JSONDecodeError as e:
            print(f"错误: JSONL文件格式错误 - {e}")
            return
        
        # 从MongoDB获取数据
        print("从MongoDB获取参考数据...")
        mongo_data = self.get_mongo_data(dataset_config)
        
        # 构建字典映射
        summary_key = dataset_config.get("summary_key", "summary")
        data_dict = {str(item["_id"]): item.get(summary_key, "") 
                    for item in mongo_data}
        
        print(f"构建了 {len(data_dict)} 个ID映射")
        
        # 生成最终结果
        final_result = []
        missing_ids = []
        
        for item in result_list:
            item_id = str(item["_id"])
            if item_id in data_dict and item["response"] != "$ERROR$":
                final_result.append({
                    "id": item["_id"], 
                    "ref_caption": data_dict[item_id], 
                    "caption": item["response"]
                })
            else:
                missing_ids.append(item_id)
        
        if missing_ids:
            print(f"警告: {len(missing_ids)} 个ID在MongoDB中未找到对应数据")
            if len(missing_ids) <= 10:  # 只显示前10个缺失的ID
                print(f"缺失的ID: {missing_ids}")
        
        # 保存结果
        output_file = dataset_config["json_save_file"]
        try:
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)
            print(f"成功保存 {len(final_result)} 条记录到: {output_file}")
        except Exception as e:
            print(f"错误: 保存文件失败 - {e}")

    def print_config_info(self):
        """打印配置信息"""
        print("=== 配置信息 ===")
        print(f"HF Endpoint: {self.config.get('hf_endpoint', '未设置')}")
        print(f"MongoDB: {self.config['mongo']['host']}:{self.config['mongo']['port']}/{self.config['mongo']['database']}")
        print(f"可用数据集: {list(self.config['datasets'].keys())}")
        print("================")


def main():
    parser = argparse.ArgumentParser(description='处理视频数据集')
    parser.add_argument('--dataset', type=str, required=True, 
                       help='要处理的数据集名称 (vista, bliss, mmavs)')
    parser.add_argument('--config', type=str, default='dataset_config.json',
                       help='配置文件路径')
    parser.add_argument('--info', action='store_true',
                       help='显示配置信息')
    
    args = parser.parse_args()
    
    try:
        processor = DatasetProcessor(args.config)
        
        if args.info:
            processor.print_config_info()
            return
        
        processor.change_jsonl(args.dataset)
        print("处理完成!")
        
    except FileNotFoundError:
        print(f"错误: 配置文件 {args.config} 不存在")
    except json.JSONDecodeError:
        print(f"错误: 配置文件 {args.config} 格式错误")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")

if __name__ == "__main__":
    main()