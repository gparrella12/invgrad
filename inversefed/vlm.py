import clip
import torch.nn.functional as F
from torchvision import transforms
import torch

    
def compute_vlm_score(reconstructed_images, reference_image, text_prompt, alpha=0.5):
    """
    Calcola un punteggio di allineamento tra l'immagine ricostruita, un'immagine di riferimento e una caption.
    
    Args:
        reconstructed_images: Tensore dell'immagine ricostruita [B, C, H, W]
        reference_image: Tensore dell'immagine di riferimento [1, C, H, W]
        text_prompt: Stringa di testo che descrive l'immagine desiderata
        alpha: Peso per bilanciare l'importanza tra allineamento al testo e all'immagine (default: 0.5)
        
    Returns:
        Punteggio complessivo di allineamento (più alto = maggior allineamento)
    """
    # Verifica se CLIP è già caricato o deve essere inizializzato
    if not hasattr(compute_vlm_score, "model"):
        device = reconstructed_images.device
        compute_vlm_score.model, compute_vlm_score.preprocess = clip.load("ViT-B/32", device=device)
        compute_vlm_score.model.eval()
    
    # Prepara le trasformazioni per adattare l'immagine al formato CLIP
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=reconstructed_images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=reconstructed_images.device).view(1, 3, 1, 1)
    
    # Funzione helper per preprocessare le immagini per CLIP
    def preprocess_for_clip(images):
        # Adatta le dimensioni dell'immagine se necessario (CLIP usa 224x224)
        if images.shape[-1] != 224 or images.shape[-2] != 224:
            images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Normalizza le immagini secondo lo standard CLIP
        if images.min() < 0:
            images = (images + 1) / 2.0  # Converti da [-1,1] a [0,1]
        
        return (images - mean) / std
    
    # Preprocessa entrambe le immagini
    reconstructed_clip = preprocess_for_clip(reconstructed_images)
    reference_clip = preprocess_for_clip(reference_image)
    
    with torch.no_grad():
        # Codifica il testo
        text_features = compute_vlm_score.model.encode_text(clip.tokenize(text_prompt).to(reconstructed_images.device))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Codifica le immagini
        reconstructed_features = compute_vlm_score.model.encode_image(reconstructed_clip)
        reconstructed_features = reconstructed_features / reconstructed_features.norm(dim=-1, keepdim=True)
        
        reference_features = compute_vlm_score.model.encode_image(reference_clip)
        reference_features = reference_features / reference_features.norm(dim=-1, keepdim=True)
    
    # Calcola la similarità coseno tra immagine ricostruita e testo (più alta = maggior allineamento)
    text_similarity = (100.0 * reconstructed_features @ text_features.T).squeeze()
    
    # Calcola la similarità coseno tra immagine ricostruita e immagine di riferimento
    image_similarity = (100.0 * reconstructed_features @ reference_features.T).squeeze()
    
    # Combina i due punteggi usando alpha come peso
    combined_score = alpha * text_similarity.mean() + (1 - alpha) * image_similarity.mean()
    
    return combined_score

# Test the function
if __name__ == "__main__":
    # Example usage
    reconstructed_images = torch.randn(1, 3, 224, 224)  # Example tensor
    reference_image = torch.randn(1, 3, 224, 224)  # Example tensor
    text_prompt = "A beautiful landscape"
    
    score = compute_vlm_score(reconstructed_images, reference_image, text_prompt, alpha=1)
    print(f"VLM Score: {score.item()}")