from collections import Counter

import clip
import requests
from PIL import Image

from Singleton import Singleton

is_mturk = True
device = "cpu"
clip_version = "RN50" if is_mturk else "ViT-L/14"

from threading import active_count
print(f"Number of active threads: {active_count()}")

@Singleton
class CLIPSingleton:
   def __init__(self):
        print('Creating CLIPSingleton...')
        print(f"Loading CLIP-{clip_version} (is_mturk: {is_mturk})...")
        model, preprocess = clip.load(clip_version, device=device)
        total_parameters = sum(p.numel() for p in model.parameters())
        print(f"Freezing CLIP params ({total_parameters})...")
        for param in model.parameters():
            param.requires_grad = False
        self.model = model
        self.preprocess = preprocess

clip_singleton = CLIPSingleton.instance() # Good. Being explicit is in line with the Python Zen
url_image_prefix = "https://gvlab-bucket.s3.amazonaws.com/"

def solve_gvlab_instance(candidates, cue, num_associations):
    clip_text = get_clip_txt(cue)
    cue_clip_txt_encoded = clip_singleton.model.encode_text(clip_text)
    cue_clip_txt_encoded /= cue_clip_txt_encoded.norm(dim=-1, keepdim=True)

    sim_for_image = {}
    for image_name in candidates:
        image_url = url_image_prefix + image_name
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        clip_image = clip_singleton.preprocess(image).unsqueeze(0)
        clip_image_feats = clip_singleton.model.encode_image(clip_image)
        clip_image_feats /= clip_image_feats.norm(dim=-1, keepdim=True)
        cue_cand_sim = get_vectors_similarity(cue_clip_txt_encoded, clip_image_feats)
        sim_for_image[image_name] = cue_cand_sim

    sorted_sim_for_image = Counter(sim_for_image).most_common()[:num_associations]
    clip_predictions = [x[0] for x in sorted_sim_for_image]
    return clip_predictions

def get_clip_txt(item):
    item = item.lower()
    vowels = ["a", "e", "i", "o", "u"]
    if any(item.startswith(x) for x in vowels):
        clip_txt = f"An {item}"
    else:
        clip_txt = f"A {item}"
    clip_txt_tokenized = clip.tokenize([clip_txt]).to(device)
    return clip_txt_tokenized

def get_vectors_similarity(v1, v2):
    similarity = v1.detach().numpy() @ v2.detach().numpy().T
    similarity_item = similarity.item()
    return similarity_item


def get_jaccard(s1, s2):
    s1 = set(s1)
    s2 = set(s2)
    jaccard = int(len(s1.intersection(s2)) / len(s1.union(s2)) * 100)
    return jaccard


def get_human_score_for_fooling_ai(data):
    candidates = data["images"]
    all_answers = []
    for ann in data["annotations"]:
        clip_predictions = solve_gvlab_instance(candidates, ann["cue"], ann["num_associations"])
        model_jaccard_score = get_jaccard(clip_predictions, ann["labels"])
        human_score = 100 - model_jaccard_score
        print(
            f"clip predictions for {ann['cue']}-{ann['num_associations']} are: {clip_predictions}. Jaccard: {model_jaccard_score}")
        ann["clip_predictions"] = clip_predictions
        ann["model_jaccard_score"] = model_jaccard_score
        ann["human_score"] = human_score
        all_answers.append(ann)
    return all_answers


if __name__ == "__main__":
    create_data = {0: {"images": ["bear.jpg", "bee.jpg", "bride.jpg", "drums.jpg", "hockey.jpg"],
    "type": "create_mturk",
   "annotations": [{"cue": "honey", "num_associations": 2, "labels": ["bear.jpg", "bee.jpg"]},
                   {"cue": "honey", "num_associations": 3, "labels": ["bear.jpg", "bee.jpg", "bride.jpg"]},
                   {"cue": "stick", "num_associations": 2, "labels": ["drums.jpg", "hockey.jpg"]},
                   {"cue": "humans", "num_associations": 2, "labels": ["bride..jpg", "hockey.jpg"]},
                   {"cue": "living creatures", "num_associations": 4, "labels": ["bear.jpg", "bee.jpg", "bride.jpg", "hockey.jpg"]}
                   ]}
                   }

    gvlab_create_instance = create_data[0]

    all_answers = get_human_score_for_fooling_ai(gvlab_create_instance)

    print(f"Aggregated answers:")
    print(all_answers)

