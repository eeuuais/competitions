# 유틸리티 함수들
def calculate_accuracy(predictions, labels):
    correct = (predictions == labels).sum().item()
    accuracy = correct / len(labels)
    return accuracy
