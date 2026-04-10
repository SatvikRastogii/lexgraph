import os
import json
import shutil

def restore_and_filter():
    corpus_dir = r"c:\Users\Satvik Rastogi\Downloads\graphrag-project\legal_corpus"
    input_dir = r"c:\Users\Satvik Rastogi\Downloads\graphrag-project\input"
    metadata_path = r"c:\Users\Satvik Rastogi\Downloads\graphrag-project\corpus_metadata.json"
    
    # Target keywords matching the 40 cases exactly as they might appear in titles
    target_keywords = {
        "maneka gandhi": False, "francis coralie": False, "olga tellis": False, "unni krishnan": False,
        "pucl": False, "people's union for civil liberties": False, "peoples union for civil liberties": False,
        "vishaka": False, "vishakha": False, "paschim banga khet mazdoor": False, 
        "mc mehta": False, "m.c. mehta": False, "m. c. mehta": False, 
        "parmanand katara": False, "dk basu": False, "d.k. basu": False, "d. k. basu": False,
        "selvi": False, "puttaswamy": False, "royappa": False, "nargesh meerza": False, "air india": False,
        "nakara": False, "ajay hasia": False, "indra sawhney": False, "anuj garg": False,
        "navtej singh": False, "navtej singh johar": False, "romesh thappar": False, 
        "brij bhushan": False, "v.g. row": False, "v. g. row": False, "sakal papers": False, 
        "bennett coleman": False, "indian express newspapers": False, "shreya singhal": False,
        "anuradha bhasin": False, "bandhua mukti morcha": False, "aruna shanbaug": False, 
        "kesavananda bharati": False, "minerva mills": False, "s.p. gupta": False, "sp gupta": False, 
        "s.r. bommai": False, "sr bommai": False, "i.r. coelho": False, "ir coelho": False, "i. r. coelho": False,
        "kihoto hollohan": False
    }

    # Wipe input directory completely
    for f in os.listdir(input_dir):
        os.remove(os.path.join(input_dir, f))
        
    print("Cleaned input/ directory.")

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    copied = 0
    copied_titles = []
    
    for doc in metadata.get('documents', []):
        title = doc.get('title', '').lower()
        filename = doc.get('filename')
        
        # Check if any target keyword is in the title
        is_target = False
        for kw in target_keywords.keys():
            if kw in title:
                is_target = True
                target_keywords[kw] = True
                break
                
        if is_target:
            source = os.path.join(corpus_dir, filename)
            dest = os.path.join(input_dir, filename)
            if os.path.exists(source):
                shutil.copy2(source, dest)
                copied += 1
                copied_titles.append(title)
            else:
                print(f"Warning: {filename} not found in legal_corpus.")

    print(f"\nRestored and extracted {copied} cases to input/!")
    if copied > 0:
        print("\nSome sample extracted cases:")
        for t in list(set(copied_titles))[:5]:
            print(f" - {t}")
            
    # Print what we missed
    missed = [kw for kw, found in target_keywords.items() if not found]
    if len(missed) > 10:
        pass # print(f"Missed keywords include: {missed[:10]}...")

if __name__ == "__main__":
    restore_and_filter()
