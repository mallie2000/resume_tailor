## model initialization
import spacy
from gliner_spacy.pipeline import GlinerSpacy
from collections import defaultdict as d

# Initialize spaCy with GLiNER
def init_model():
    model = spacy.blank("en")
    model.add_pipe("gliner_spacy", config={
        "gliner_model": "urchade/gliner_mediumv2.1",
        "chunk_size": 250,
        "labels": [
            "programming language", "framework", "database", "cloud platform",
            "version control", "methodology",
            "tool", "soft skill"
        ],
        "style": "ent",
        "threshold": 0.3,
        "map_location": "cpu"
    })
    return model

def get_jobs_desc(filePath = "job.txt"):
    with open(f"{filePath}",'r') as f:
        text = f.read()
        text = text.replace('\n',' ')
    return text

def extract_skills(model,text):
    skill_dict = d(set)
    doc = model(text)
    for ent in doc.ents:
        skill_dict[ent.label_].add(ent.text)
    return skill_dict

def main():
    model = init_model()
    text = get_jobs_desc()         
    skills = extract_skills(model,text)

    for key in skills.keys():
        print(f"{key} : {skills[key]}")
        
if __name__ == "__main__":
    main()