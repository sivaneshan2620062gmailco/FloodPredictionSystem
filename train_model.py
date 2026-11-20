from src.model import train_model

if __name__ == "__main__":
    _, accuracy = train_model()
    print(f"Model test accuracy: {accuracy:.3f}")
