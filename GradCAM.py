import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None    # To store gradients
        self.activations = None  # To store activations

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_hook(self, module, input, output):
        self.activations = output
        output.register_hook(self.save_gradient)  # Register the gradient hook

    def __call__(self, x, class_idx=None):
        # Register forward hook on the target layer
        target_layer = dict(self.model.named_modules())[self.target_layer]
        handle = target_layer.register_forward_hook(self.forward_hook)

        # Forward pass
        output = self.model(x)
        if class_idx is None:
            class_idx = torch.argmax(output)  # Use the predicted class if not specified
        loss = output[:, class_idx]

        # Backward pass
        self.model.zero_grad()
        loss.backward()

        # Remove the hook
        handle.remove()

        # Calculate Grad-CAM
        grads = self.gradients.detach().cpu().numpy()[0]  # Gradients
        activations = self.activations.detach().cpu().numpy()[0]  # Activations
        weights = np.mean(grads, axis=(1, 2))  # Global average pooling over the gradients

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)  # ReLU to keep only positive values
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

def get_misclassified_images(neural_net, test_loader, device):
    """ Function to get misclassified images """
    neural_net.eval()
    misclassified = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = neural_net(inputs)
            preds = torch.argmax(outputs, dim=1)
            for i in range(len(inputs)):
                if preds[i] != labels[i]:
                    misclassified.append((inputs[i].cpu(), labels[i].item(), preds[i].item()))
    return misclassified

def show_cam_on_image(img: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap (mask) on an image.

    Args:
        img (np.ndarray): Original image (H, W, C) in the range [0, 1].
        mask (np.ndarray): Grad-CAM heatmap (H, W) in the range [0, 1].
        alpha (float): Blending factor for overlay.

    Returns:
        np.ndarray: Image with Grad-CAM heatmap overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)  # Convert mask to heatmap
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    overlay = heatmap / 255.0 + img  # Combine heatmap with image
    overlay = overlay / overlay.max()  # Normalize to range [0, 1]

    return np.uint8(255 * overlay)  # Convert to range [0, 255]

def visualize_misclassified_with_gradcam(neural_net, misclassified, grad_cam, target_layer):
    """ Visualize misclassified images with Grad-CAM"""
    neural_net.eval()  # Ensure model is in evaluation mode
    for i, (img, true_label, pred_label) in enumerate(misclassified):
        # Prepare image for Grad-CAM
        img_input = img.unsqueeze(0).to(dev)                    # Add batch dimension
        mask = grad_cam(img_input, class_idx=pred_label)        # Grad-CAM for predicted label

        # Normalize Grad-CAM mask
        mask = cv2.resize(mask, (img.shape[1], img.shape[2]))   # Resize mask to match image
        mask = (mask - mask.min()) / (mask.max() - mask.min())  # Normalize [0, 1]

        # Convert image for visualization
        img_np = img.permute(1, 2, 0).numpy()  # Convert to HWC
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # Normalize [0, 1]
        img_np = cv2.resize(img_np, (mask.shape[1], mask.shape[0]))       # Match dimensions

        # Create overlay
        overlay = show_cam_on_image(img_np, mask, alpha=0.6)

        # Plot image, overlay, and labels
        plt.figure(figsize=(6, 3))  # Increase figure size
        plt.subplot(1, 2, 1)
        plt.imshow(img_np)
        plt.title(f"True: {true_label}, Pred: {pred_label}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(overlay)
        plt.title("Grad-CAM")
        plt.axis('off')

        plt.tight_layout()
        plt.show()