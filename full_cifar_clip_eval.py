# 置顶环境变量
import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 规范导入
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# CIFAR-10 标签
CIFAR10_LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class CIFAR10Dataset(Dataset):
    """CIFAR-10 测试集数据集类"""
    
    def __init__(self, pickle_path):
        """初始化数据集"""
        with open(pickle_path, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
        
        self.images = data_dict[b'data']
        self.labels = data_dict[b'labels']
    
    def __len__(self):
        """返回数据集长度"""
        return len(self.labels)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        # 读取原始 3072 字节的向量
        img_data = self.images[idx]
        # 转换为 (32, 32, 3) 的 NumPy 数组
        img = self.convert_to_image(img_data)
        # 返回原始的 uint8 格式的 NumPy 数组
        label = self.labels[idx]
        return img, label
    
    def convert_to_image(self, img_data):
        """将原始向量转换为图像"""
        # reshape 为 (3, 32, 32)
        img = img_data.reshape(3, 32, 32)
        # transpose 为 (32, 32, 3)
        img = img.transpose(1, 2, 0)
        # 保持原始的 uint8 格式 (0-255)
        img = img.astype(np.uint8)
        return img

def load_model():
    """加载 CLIP 模型和处理器"""
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        use_safetensors=True,
        local_files_only=False
    )
    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32",
        use_safetensors=True,
        local_files_only=False
    )
    # 移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, processor, device

def batch_inference(model, processor, dataloader, device, labels):
    """批量进行推理"""
    all_preds = []  # 保存所有预测结果
    all_labels = []  # 保存所有真实标签
    wrong_predictions = []
    
    # 优化 Prompt 模板
    prompt_labels = ["a photo of a " + label for label in labels]
    print(f"使用的 Prompt 模板: {prompt_labels}")
    
    # 性能优化：在循环外只计算一次文本输入，避免重复处理
    text_inputs = processor(text=prompt_labels, return_tensors="pt", padding=True).to(device)
    text_input_ids = text_inputs['input_ids']
    text_attention_mask = text_inputs['attention_mask']
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, true_labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
            # 处理图像输入 - 让处理器自动进行预处理，输入是 NumPy Batch
            image_inputs = processor(images=images, return_tensors="pt").to(device)
            
            # 模型推理 - 使用标准参数名
            outputs = model(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
                pixel_values=image_inputs['pixel_values']
            )
            
            # 获取 logits 并计算概率
            logits_per_image = outputs.logits_per_image  # 形状: [batch_size, num_labels]
            probs = logits_per_image.softmax(dim=-1)
            
            # 获取预测结果 - 在最后一个维度上进行 argmax
            predictions = probs.argmax(dim=-1).cpu().numpy()
            
            # 收集结果 - 确保累加所有批次的结果
            all_preds.extend(predictions)
            all_labels.extend(true_labels.numpy())
            
            # 记录错误预测
            if len(wrong_predictions) < 5:
                for i, (pred, true) in enumerate(zip(predictions, true_labels.numpy())):
                    if pred != true:
                        wrong_predictions.append((true, pred))
                        if len(wrong_predictions) >= 5:
                            break
    
    # 确保返回全量数据
    return np.array(all_preds), np.array(all_labels), wrong_predictions

def calculate_accuracy(predictions, labels):
    """计算准确率"""
    correct = (predictions == labels).sum()
    total = len(labels)
    accuracy = correct / total
    return accuracy

def plot_confusion_matrix(predictions, labels, class_names, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(labels, predictions)
    
    # 计算百分比
    cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Percentage)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"混淆矩阵已保存到: {save_path}")

def main():
    """主函数"""
    try:
        # 数据路径 - 使用相对路径
        pickle_path = "./cifar-10-batches-py/test_batch"
        
        # 打印当前目录和文件路径
        print("=== 开始执行 CLIP 模型评估 ===")
        print(f"当前工作目录: {os.getcwd()}")
        print(f"数据文件路径: {pickle_path}")
        print(f"文件是否存在: {os.path.exists(pickle_path)}")
        
        # 检查文件是否存在
        if not os.path.exists(pickle_path):
            print(f"错误: 文件 {pickle_path} 不存在！")
            return
        
        # 超参数
        batch_size = 32
        
        # 加载数据集
        print("加载 CIFAR-10 测试集...")
        dataset = CIFAR10Dataset(pickle_path)
        print(f"数据集大小: {len(dataset)}")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        print(f"数据加载器批次数量: {len(dataloader)}")
        
        # 测试数据加载
        print("测试数据加载器...")
        for batch_idx, (images, true_labels) in enumerate(dataloader):
            print(f"批次 {batch_idx}: 图像形状: {images.shape}, 标签形状: {true_labels.shape}")
            # 只测试第一个批次
            break
        
        # 加载模型
        print("加载 CLIP 模型...")
        try:
            model, processor, device = load_model()
            print(f"使用设备: {device}")
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return
        
        # 批量推理
        print("开始批量推理...")
        try:
            all_preds, all_labels, wrong_predictions = batch_inference(
                model, processor, dataloader, device, CIFAR10_LABELS
            )
            print(f"推理完成，预测数量: {len(all_preds)}")
            print(f"真实标签数量: {len(all_labels)}")
        except Exception as e:
            print(f"推理时出错: {e}")
            return
        
        # 计算准确率
        accuracy = calculate_accuracy(all_preds, all_labels)
        print(f"Top-1 准确率: {accuracy:.4f}")
        
        # 生成混淆矩阵
        print("生成混淆矩阵...")
        confusion_matrix_path = "confusion_matrix.png"
        plot_confusion_matrix(all_preds, all_labels, CIFAR10_LABELS, confusion_matrix_path)
        
        # 打印错误预测
        print("\n前 5 个预测错误的案例:")
        for i, (true_idx, pred_idx) in enumerate(wrong_predictions[:5]):
            true_label = CIFAR10_LABELS[true_idx]
            pred_label = CIFAR10_LABELS[pred_idx]
            print(f"案例 {i+1}: 真实标签: {true_label}, 预测标签: {pred_label}")
        
        print("\n=== 评估完成 ===")
    except Exception as e:
        print(f"执行过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
