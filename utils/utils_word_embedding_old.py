import torch
import numpy as np
import fasttext.util
from gensim import models

def verify_glove_words(filename='./utils/glove.6B.300d.txt', words_to_check=['dark', 'soft', 'fog', 'mist', 'low', 'rain', 'snow', 'clear']):
    """Verify that certain words exist in the GloVe vocabulary."""
    found_words = set()
    with open(filename, 'rb') as f:
        for line in f:
            word = line.decode().strip().split(' ')[0]
            if word in words_to_check:
                found_words.add(word)
                if len(found_words) == len(words_to_check):
                    break
    
    print("\nGloVe vocabulary verification:")
    for word in words_to_check:
        status = "✓" if word in found_words else "✗"
        print(f"{status} {word}")
    
    return found_words

def load_word_embeddings(emb_file, vocab):
    embeds = {}
    for line in open(emb_file, 'rb'):
        line = line.decode().strip().split(' ')
        wvec = torch.FloatTensor(list(map(float, line[1:])))
        embeds[line[0]] = wvec

    # for zappos (should account for everything)
    custom_map = {
        'Faux.Fur':'fake_fur', 'Faux.Leather':'fake_leather', 'Full.grain.leather':'thick_leather', 
        'Hair.Calf':'hair_leather', 'Patent.Leather':'shiny_leather', 'Nubuck':'grainy_leather', 
        'Boots.Ankle':'ankle_boots', 'Boots.Knee.High':'knee_high_boots', 'Boots.Mid-Calf':'midcalf_boots', 
        'Shoes.Boat.Shoes':'boat_shoes', 'Shoes.Clogs.and.Mules':'clogs_shoes', 'Shoes.Flats':'flats_shoes',
        'Shoes.Heels':'heels', 'Shoes.Loafers':'loafers', 'Shoes.Oxfords':'oxford_shoes',
        'Shoes.Sneakers.and.Athletic.Shoes':'sneakers'}
    custom_map_vaw = {
        'selfie': 'photo'
    }

    E = []
    for k in vocab:
        if k in custom_map:
            print(f'Change {k} to {custom_map[k]}')
            k = custom_map[k]
        k = k.lower()
        if '_' in k:
            toks = k.split('_')
            emb_tmp = torch.zeros(300).float()
            for tok in toks:
                if tok in custom_map_vaw:
                    tok = custom_map_vaw[tok]
                emb_tmp += embeds[tok]
            emb_tmp /= len(toks)
            E.append(emb_tmp)
        else:
            E.append(embeds[k])

    embeds = torch.stack(E)
    print ('Loaded embeddings from file %s' % emb_file, embeds.size())

    return embeds

def load_fasttext_embeddings(emb_file,vocab):
    custom_map = {
        'Faux.Fur': 'fake fur',
        'Faux.Leather': 'fake leather',
        'Full.grain.leather': 'thick leather',
        'Hair.Calf': 'hairy leather',
        'Patent.Leather': 'shiny leather',
        'Boots.Ankle': 'ankle boots',
        'Boots.Knee.High': 'kneehigh boots',
        'Boots.Mid-Calf': 'midcalf boots',
        'Shoes.Boat.Shoes': 'boatshoes',
        'Shoes.Clogs.and.Mules': 'clogs shoes',
        'Shoes.Flats': 'flats shoes',
        'Shoes.Heels': 'heels',
        'Shoes.Loafers': 'loafers',
        'Shoes.Oxfords': 'oxford shoes',
        'Shoes.Sneakers.and.Athletic.Shoes': 'sneakers',
        'traffic_light': 'traficlight',
        'trash_can': 'trashcan',
        'dry-erase_board' : 'dry_erase_board',
        'black_and_white' : 'black_white',
        'eiffel_tower' : 'tower'
    }
    vocab_lower = [v.lower() for v in vocab]
    vocab = []
    for current in vocab_lower:
        if current in custom_map:
            vocab.append(custom_map[current])
        else:
            vocab.append(current)

    
    ft = fasttext.load_model(emb_file) #DATA_FOLDER+'/fast/cc.en.300.bin')
    embeds = []
    for k in vocab:
        if '_' in k:
            ks = k.split('_')
            emb = np.stack([ft.get_word_vector(it) for it in ks]).mean(axis=0)
        else:
            emb = ft.get_word_vector(k)
        embeds.append(emb)

    embeds = torch.Tensor(np.stack(embeds))
    print('Fasttext Embeddings loaded, total embeddings: {}'.format(embeds.size()))
    return embeds

def load_word2vec_embeddings(emb_file,vocab):
    # vocab = [v.lower() for v in vocab]

    
    model = models.KeyedVectors.load_word2vec_format(emb_file,binary=True)
        #DATA_FOLDER+'/w2v/GoogleNews-vectors-negative300.bin', binary=True)

    custom_map = {
        'Faux.Fur': 'fake_fur',
        'Faux.Leather': 'fake_leather',
        'Full.grain.leather': 'thick_leather',
        'Hair.Calf': 'hair_leather',
        'Patent.Leather': 'shiny_leather',
        'Boots.Ankle': 'ankle_boots',
        'Boots.Knee.High': 'knee_high_boots',
        'Boots.Mid-Calf': 'midcalf_boots',
        'Shoes.Boat.Shoes': 'boat_shoes',
        'Shoes.Clogs.and.Mules': 'clogs_shoes',
        'Shoes.Flats': 'flats_shoes',
        'Shoes.Heels': 'heels',
        'Shoes.Loafers': 'loafers',
        'Shoes.Oxfords': 'oxford_shoes',
        'Shoes.Sneakers.and.Athletic.Shoes': 'sneakers',
        'traffic_light': 'traffic_light',
        'trash_can': 'trashcan',
        'dry-erase_board' : 'dry_erase_board',
        'black_and_white' : 'black_white',
        'eiffel_tower' : 'tower'
    }

    embeds = []
    for k in vocab:
        if k in custom_map:
            k = custom_map[k]
        if '_' in k and k not in model:
            ks = k.split('_')
            emb = np.stack([model[it] for it in ks]).mean(axis=0)
        else:
            emb = model[k]
        embeds.append(emb)
    embeds = torch.Tensor(np.stack(embeds))
    print('Word2Vec Embeddings loaded, total embeddings: {}'.format(embeds.size()))
    return embeds


def initialize_wordembedding_matrix(embedding_type, type_name):
    """Initialize word embeddings for weather condition types."""
    if embedding_type == 'glove':
        # Load raw word embeddings
        word_embeddings = {}
        print("\nLoading GloVe embeddings...")
        
        try:
            with open('./utils/glove.6B.300d.txt', 'rb') as f:
                # First, let's check the first few words to debug
                print("\nFirst 20 words in GloVe vocabulary:")
                for i, line in enumerate(f):
                    if i < 20:  # Print first 20 words
                        word = line.decode().strip().split(' ')[0]
                        print(f"  {word}")
                    line = line.decode().strip().split(' ')
                    word_embeddings[line[0]] = torch.FloatTensor(list(map(float, line[1:])))
        except FileNotFoundError:
            print("\nError: GloVe file not found!")
            print("Expected path: ./utils/glove.6B.300d.txt")
            print("Current working directory:", os.getcwd())
            raise
        
        embedding_dim = 300
        weight = torch.zeros(len(type_name), embedding_dim)
        
        print(f"\nInitializing word embeddings for {len(type_name)} classes")
        
        # Let's check what common words are available
        test_words = ['the', 'a', 'an', 'water', 'air', 'clear', 'low', 'rain', 'snow',
                     'wet', 'dry', 'cold', 'warm', 'light', 'heavy', 'dark', 'bright',
                     'fog', 'mist', 'cloud', 'weather', 'storm', 'wind']
        
        print("\nChecking common words availability:")
        available_words = {}
        for word in test_words:
            if word in word_embeddings:
                available_words[word] = True
                print(f"  ✓ {word}")
            else:
                available_words[word] = False
                print(f"  ✗ {word}")
        
        # Use only available words for embeddings
        word_mapping = {}
        for word in type_name:
            if word in word_embeddings:
                word_mapping[word] = [word]
            elif word == 'blur':
                word_mapping[word] = ['light']  # we'll update this based on available words
            elif word == 'haze':
                word_mapping[word] = ['light']  # we'll update this based on available words
        
        print("\nWord mappings:")
        for original, mapped in word_mapping.items():
            print(f"  {original} -> {mapped}")
        
        # Now process each type with the available words
        for i, type_i in enumerate(type_name):
            try:
                words = type_i.split('_') if '_' in type_i else [type_i]
                vec = torch.zeros(embedding_dim)
                word_count = 0
                
                print(f"\nProcessing {type_i}...")
                for word in words:
                    mapped_words = word_mapping.get(word, [word])
                    for mapped_word in mapped_words:
                        if mapped_word in word_embeddings:
                            vec += word_embeddings[mapped_word]
                            word_count += 1
                            print(f"  Using embedding for '{mapped_word}'")
                
                if word_count > 0:
                    vec /= word_count
                else:
                    print(f"  No valid embeddings found for '{type_i}', using random initialization")
                    vec = torch.randn(embedding_dim) * 0.02
                
                weight[i] = vec
                
            except Exception as e:
                print(f"Error processing '{type_i}': {str(e)}")
                weight[i] = torch.randn(embedding_dim) * 0.02
        
        print(f"\nWord embedding matrix shape: {weight.shape}")
        return weight, embedding_dim
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")
    
# def initialize_wordembedding_matrix(embedding_type, type_name):
#     if embedding_type == 'glove':
#         word_embeddings = load_word_embeddings('./utils/glove.6B.300d.txt')
#         embedding_dim = 300
#         weight = torch.zeros(len(type_name), embedding_dim)
        
#         # Print debugging info
#         print(f"Initializing word embeddings for {len(type_name)} classes")
#         print("Classes:", type_name)
        
#         for i, type_i in enumerate(type_name):
#             try:
#                 if '_' in type_i:  # Handle compound words
#                     words = type_i.split('_')
#                     vec = sum(word_embeddings[word] for word in words) / len(words)
#                 else:
#                     vec = word_embeddings[type_i]
#                 weight[i] = torch.FloatTensor(vec)
#             except KeyError as e:
#                 print(f"Warning: Word '{type_i}' not found in embeddings, using random initialization")
#                 weight[i] = torch.randn(embedding_dim) * 0.02  # Random initialization
                
#         return weight, embedding_dim

# def initialize_wordembedding_matrix(name, vocab):
#     """
#     Args:
#     - name: hyphen separated word embedding names: 'glove-word2vec-conceptnet'.
#     - vocab: list of attributes/objects.
#     """
#     wordembs = name.split('+')
#     result = None

#     for wordemb in wordembs:
#         if wordemb == 'glove':
#             wordemb_ = load_word_embeddings(f'./utils/glove.6B.300d.txt', vocab)
#         if result is None:
#             result = wordemb_
#         else:
#             result = torch.cat((result, wordemb_), dim=1)
#     dim = 300 * len(wordembs)
#     return result, dim
