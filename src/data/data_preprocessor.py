import os
import json
import pickle
import shutil
from typing import List, Dict, Tuple, Generator
from src.b2e.utils import utils

class DataPreprocessor:
    def __init__(self, config: Dict):
        self.config = config
        self.txs_path = self.config['txsPath']
        self.data_path = self.config['dataPath']
        self.raw_data_path = os.path.join(self.data_path, 'raw_data')
        self.processed_data_path = os.path.join(self.data_path, 'processed_data')
        self.broker_balance_path = self.config['brokerBalancePath']
        self.shard_num = self.config['shardNum']
        self.txs_per_epoch = self.config['txsPerEpoch']

    def process_file_data(self, start_file_idx: int = 0, remaining_txs: List[str] = None) -> Generator[List[str], None, None]:
        """处理文件数据，支持跨文件的epoch
        Args:
            start_file_idx: 开始处理的文件索引
            remaining_txs: 上一个文件剩余的交易
        """
        current_batch = remaining_txs if remaining_txs else []
        
        for file_idx in range(start_file_idx, len(self.txs_path)):
            print(f"Processing file: {self.txs_path[file_idx]}")
            with open(self.txs_path[file_idx], 'r', buffering=8192*1024) as f:
                next(f)  # 跳过header
                
                for line in f:
                    current_batch.append(line)
                    
                    if len(current_batch) == self.txs_per_epoch:
                        yield current_batch
                        current_batch = []
            
            # 如果这是最后一个文件，并且还有剩余交易，作为最后一个epoch
            if file_idx == len(self.txs_path) - 1 and current_batch:
                yield current_batch
            # 如果不是最后一个文件，保留剩余交易，等待下一个文件补充
            elif current_batch:
                continue

    def save_epochs_chunk(self, epochs_data: List[Dict], chunk_id: int):
        """保存一组epoch数据到pkl文件"""
        chunk_file = os.path.join(self.raw_data_path, f'raw_epochs_chunk_{chunk_id}.pkl')
        with open(chunk_file, 'wb') as f:
            pickle.dump(epochs_data, f)

    def load_epochs_chunk(self, chunk_id: int) -> List[Dict]:
        """加载指定的epoch数据块"""
        chunk_file = os.path.join(self.raw_data_path, f'raw_epochs_chunk_{chunk_id}.pkl')
        if not os.path.exists(chunk_file):
            raise FileNotFoundError(f"Chunk file {chunk_file} not found")
        
        with open(chunk_file, 'rb') as f:
            return pickle.load(f)

    def get_epoch_data(self, epoch: int, chunk_size: int = 200) -> Dict:
        """获取指定epoch的数据"""
        chunk_id = epoch // chunk_size
        epoch_in_chunk = epoch % chunk_size
        
        chunk_data = self.load_epochs_chunk(chunk_id)
        return chunk_data[epoch_in_chunk]

    def read_transactions(self, data: List[str]) -> List[Tuple]:
        """
        读取交易数据并转换为内部格式
        返回的交易格式为: (id, from_address, to_address, value, fee)
        """
        txs = []
        for i, line in enumerate(data):
            line = line.strip().split(",")
            if line[13] != "None" or line[7] == "1":
                continue
            tx = (i, line[3], line[4], int(line[8]), eval(line[10]) * int(line[11]))
            txs.append(tx)
        return txs

    def process_batch_data(self, k_number: int, shard_num: int, tx_batch: List[str], capacities: List[int]) -> Tuple[List, List, List]:
        """处理一批交易数据"""
        txs = self.read_transactions(tx_batch)
        ctxs = []
        
        for tx in txs:
            if tx[1] == "None" or tx[2] == "None":
                continue
            if utils.Addr2Shard(tx[1], shard_num) != utils.Addr2Shard(tx[2], shard_num) and tx[3] != 0:
                ctxs.append([tx, utils.Addr2Shard(tx[1], shard_num), utils.Addr2Shard(tx[2], shard_num)])
        
        new_ctxs = [ctx for ctx in ctxs if int(ctx[0][-2]) > 10]
        return txs, new_ctxs, capacities

    def generate_epoch_data(self, tx_batch: List[str], epoch: int) -> Dict:
        """生成单个epoch的数据"""
        with open(self.broker_balance_path, "r") as f:
            capacities = [int(i.strip()) for i in f.readlines()]
        
        k_number = len(capacities)
        txs, ctxs, capacities = self.process_batch_data(
            k_number, 
            self.shard_num, 
            tx_batch, 
            capacities
        )
        
        return {
            'txs': txs,
            'ctxs': ctxs,
            'capacities': capacities,
            'epoch': epoch,
            'total_tx_count': len(tx_batch),
            'valid_tx_count': len(txs)
        }

    def preprocess_raw_data(self, num_epochs: int = 1000, chunk_size: int = 200):
        """分批预处理原始交易数据"""
        if os.path.exists(os.path.join(self.raw_data_path, 'raw_epochs_chunk_0.pkl')):
            print("Raw data already exists. Skipping preprocessing.")
            return
            
        os.makedirs(self.raw_data_path, exist_ok=True)
        
        current_epoch = 0
        current_chunk = []
        chunk_id = 0
        remaining_txs = None
        
        print("Starting preprocessing...")
        for file_idx in range(len(self.txs_path)):
            for tx_batch in self.process_file_data(file_idx, remaining_txs):
                if current_epoch >= num_epochs:
                    break
                if(current_epoch % 50 == 0):
                    print(f"Processing epoch {current_epoch}, batch size: {len(tx_batch)}")
                epoch_data = self.generate_epoch_data(tx_batch, current_epoch)
                current_chunk.append(epoch_data)
                
                if len(current_chunk) >= chunk_size:
                    print(f"Saving chunk {chunk_id} with {len(current_chunk)} epochs")
                    self.save_epochs_chunk(current_chunk, chunk_id)
                    chunk_id += 1
                    current_chunk = []
                
                current_epoch += 1
                
                # 获取剩余的交易用于下一个文件
                if len(tx_batch) < self.txs_per_epoch:
                    remaining_txs = tx_batch
                else:
                    remaining_txs = None
        
        # 保存最后剩余的chunk
        if current_chunk:
            print(f"Saving final chunk {chunk_id} with {len(current_chunk)} epochs")
            self.save_epochs_chunk(current_chunk, chunk_id)
            
        print(f"Preprocessing completed. Total chunks: {chunk_id + 1}")

    def create_experiment(self, experiment_name: str, num_epochs: int, use_same_data: bool = False, clear_data: bool = False, chunk_size: int = 200):
        """创建新的实验数据集"""
        experiment_path = os.path.join(self.processed_data_path, experiment_name)
        
        # 检查是否存在预处理的数据
        if not os.path.exists(os.path.join(self.raw_data_path, 'raw_epochs_chunk_0.pkl')):
            raise FileNotFoundError("No preprocessed data found. Please run preprocess_raw_data first.")
            
        if clear_data and os.path.exists(experiment_path):
            shutil.rmtree(experiment_path)
        os.makedirs(experiment_path, exist_ok=True)

        # 获取可用的预处理数据总数
        available_chunks = len([f for f in os.listdir(self.raw_data_path) 
                              if f.startswith('raw_epochs_chunk_') and f.endswith('.pkl')])
        total_available_epochs = available_chunks * chunk_size

        # 创建进度显示
        total_work = num_epochs if not use_same_data else 1
        print(f"Creating experiment with {total_work} epochs to process...")

        if use_same_data:
            # 只获取第一个epoch的数据
            base_epoch = self.get_epoch_data(0, chunk_size)
            self.save_epoch_data(base_epoch, experiment_path, 0)
            print("Base epoch data saved, creating copies...")
            
            # 复制到其他epochs
            base_epoch_dir = os.path.join(experiment_path, 'item0')
            for epoch in range(1, num_epochs):
                dst = os.path.join(experiment_path, f'item{epoch}')
                if not os.path.exists(dst):
                    shutil.copytree(base_epoch_dir, dst)
                if epoch % 100 == 0:
                    print(f"Copied {epoch}/{num_epochs} epochs...")
        else:
            for epoch in range(num_epochs):
                if not os.path.exists(os.path.join(experiment_path, f'item{epoch}')):
                    # 计算要使用的实际epoch索引（循环使用可用数据）
                    epoch_to_use = epoch % total_available_epochs
                    try:
                        epoch_data = self.get_epoch_data(epoch_to_use, chunk_size)
                        self.save_epoch_data(epoch_data, experiment_path, epoch)
                    except Exception as e:
                        print(f"Warning: Failed to process epoch {epoch}: {str(e)}")
                        continue
                
                if epoch % 50 == 0:
                    print(f"Processed {epoch}/{num_epochs} epochs...")

        print(f"Successfully created/updated experiment '{experiment_name}' with {num_epochs} epochs")

    def save_epoch_data(self, epoch_data: Dict, experiment_path: str, epoch: int):
        """
        保存单个epoch的数据
        ctx格式: [transaction, source_shard, destination_shard]
        其中 transaction 格式为: (id, from_address, to_address, value, fee)
        """
        epoch_dir = os.path.join(experiment_path, f'item{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        # 使用原始交易总数计算切片范围
        slice_start = epoch * self.txs_per_epoch
        slice_end = slice_start + epoch_data['total_tx_count']
        
        # 保存data1.txt
        with open(os.path.join(epoch_dir, 'data1.txt'), 'w') as f:
            f.write(f"Knapsack number : {len(epoch_data['capacities'])}\n")
            f.write(f"Slice : {slice_start},{slice_end}\n")
            f.write(f"Ctxs : {len(epoch_data['ctxs'])}\n")
            for ctx in epoch_data['ctxs']:
                f.write(f"({ctx[0][-1]},{ctx[0][-2]},{ctx[1]},{ctx[2]}) ")
            f.write("\n")
            f.write(", ".join(map(str, epoch_data['capacities'])))

        # 保存Ctx.csv
        with open(os.path.join(epoch_dir, "Ctx.csv"), "w") as f:
            f.write("id,Tx sour.shard,Tx dest.shard,Fee,Value,Ratio\n")
            for index, ctx in enumerate(epoch_data['ctxs']):
                f.write(f"{index},{ctx[1]},{ctx[2]},{ctx[0][-1]},{ctx[0][-2]},{ctx[0][-1]/ctx[0][-2]}\n")