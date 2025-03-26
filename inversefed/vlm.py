import clip
import torch.nn.functional as F
from torchvision import transforms
import torch

    
def compute_vlm_score(images, text_prompt):
    """
    Calcola un punteggio di allineamento tra immagini e testo usando un modello VLM (CLIP).
    
    Args:
        images: Tensore delle immagini normalizzate [B, C, H, W]
        text_prompt: Stringa di testo che descrive l'immagine desiderata
        
    Returns:
        Punteggio di perdita (più basso = maggior allineamento)
    """
    # Importiamo le librerie necessarie per CLIP

    
    # Verifica se CLIP è già caricato o deve essere inizializzato
    if not hasattr(compute_vlm_score, "model"):
        device = images.device
        compute_vlm_score.model, compute_vlm_score.preprocess = clip.load("ViT-B/32", device=device)
        compute_vlm_score.model.eval()
    
    # Prepara le trasformazioni per adattare l'immagine al formato CLIP
    # Le immagini potrebbero essere già normalizzate secondo mean/std diversi
    # Quindi denormalizziamo e poi rinormalizziamo secondo gli standard di CLIP
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device).view(1, 3, 1, 1)
    
    # Adatta le dimensioni dell'immagine se necessario (CLIP usa 224x224)
    if images.shape[-1] != 224 or images.shape[-2] != 224:
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
    
    # Normalizza le immagini secondo lo standard CLIP
    # Assumiamo che le immagini siano già in intervallo [0,1] o lo portiamo in tale intervallo
    if images.min() < 0:
        images = (images + 1) / 2.0  # Converti da [-1,1] a [0,1]
    
    images_clip = (images - mean) / std
    
    # Codifica il testo
    with torch.no_grad():
        text_features = compute_vlm_score.model.encode_text(clip.tokenize(text_prompt).to(images.device))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Codifica l'immagine
        image_features = compute_vlm_score.model.encode_image(images_clip)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # Calcola la similarità coseno (più alta = maggior allineamento)
    similarity = (100.0 * image_features @ text_features.T).squeeze()
    
    # Restituisci il punteggio di perdita (più basso = maggior allineamento)
    # Usiamo il negativo della similarità per avere una loss
    return similarity.mean()