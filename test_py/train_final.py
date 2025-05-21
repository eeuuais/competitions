{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c8929eaa",
   "metadata": {},
   "outputs": [
    {
     "ename": "ImportError",
     "evalue": "cannot import name 'SimpleDataset' from 'dataset' (c:\\Users\\09048\\2025project\\competitions\\test_py\\dataset.py)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mImportError\u001b[0m                               Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[5], line 7\u001b[0m\n\u001b[0;32m      4\u001b[0m sys\u001b[38;5;241m.\u001b[39mpath\u001b[38;5;241m.\u001b[39mappend(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m./\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[0;32m      6\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mmodel\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m SimpleModel  \u001b[38;5;66;03m# model.py에서 SimpleModel 클래스를 import\u001b[39;00m\n\u001b[1;32m----> 7\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mdataset\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m SimpleDataset  \u001b[38;5;66;03m# dataset.py에서 SimpleDataset 클래스를 import\u001b[39;00m\n\u001b[0;32m      8\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mutils\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m calculate_accuracy  \u001b[38;5;66;03m# utils.py에서 calculate_accuracy 함수 import\u001b[39;00m\n\u001b[0;32m     11\u001b[0m \u001b[38;5;66;03m# 모델 초기화\u001b[39;00m\n",
      "\u001b[1;31mImportError\u001b[0m: cannot import name 'SimpleDataset' from 'dataset' (c:\\Users\\09048\\2025project\\competitions\\test_py\\dataset.py)"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "from model import SimpleModel\n",
    "from dataset import SimpleDataset\n",
    "from utils import calculate_accuracy\n",
    "\n",
    "# 모델 초기화\n",
    "model = SimpleModel()\n",
    "\n",
    "# 데이터셋 준비 (예시로 랜덤 데이터 사용)\n",
    "train_features = torch.randn(100, 784)  # 100개의 샘플, 784개의 특징\n",
    "train_labels = torch.randint(0, 10, (100,))  # 100개의 샘플, 0~9까지의 라벨\n",
    "\n",
    "train_dataset = SimpleDataset(train_features, train_labels)\n",
    "train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)\n",
    "\n",
    "# 손실 함수와 옵티마이저\n",
    "criterion = torch.nn.CrossEntropyLoss()\n",
    "optimizer = torch.optim.Adam(model.parameters(), lr=0.001)\n",
    "\n",
    "# 학습 루프\n",
    "for epoch in range(10):  # 10 epochs\n",
    "    model.train()\n",
    "    total_loss = 0\n",
    "    correct = 0\n",
    "    total = 0\n",
    "    for inputs, labels in train_loader:\n",
    "        optimizer.zero_grad()\n",
    "        outputs = model(inputs)\n",
    "        loss = criterion(outputs, labels)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "\n",
    "        total_loss += loss.item()\n",
    "        _, predicted = torch.max(outputs, 1)\n",
    "        correct += (predicted == labels).sum().item()\n",
    "        total += labels.size(0)\n",
    "\n",
    "    accuracy = correct / total\n",
    "    print(f\"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
