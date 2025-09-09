import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def pair_features(sent1, sent2, model, variant="diff", is_use=False):
    """
    Generate features for a pair of sentences based on variant:
      - 'diff'     : |e1 - e2|              (d)
      - 'concat3'  : [e1, e2, |e1-e2|]      (3d)
      - 'concatcos': [e1, e2, cos(e1,e2)]   (2d+1)
      - 'concat'   : [e1, e2]               (2d)
      - 'cos'      : [cos(e1, e2)]          (1)
    """
    if is_use:
        e1 = model([sent1]).numpy()[0]
        e2 = model([sent2]).numpy()[0]
    else:
        e1 = model.encode(sent1)
        e2 = model.encode(sent2)

    if variant == "diff":
        return np.abs(e1 - e2)

    elif variant == "concat3":
        return np.concatenate([e1, e2, np.abs(e1 - e2)])

    elif variant == "concatcos":
        cos = cosine_similarity(e1.reshape(1,-1), e2.reshape(1,-1))[0][0]
        return np.concatenate([e1, e2, [cos]])

    elif variant == "concat":
        return np.concatenate([e1, e2])

    elif variant == "cos":
        cos = cosine_similarity(e1.reshape(1,-1), e2.reshape(1,-1))[0][0]
        return np.array([cos])

    else:
        raise ValueError(f"Unknown variant: {variant}")

def embed_pairs_variant(pairs, model, variant="diff", is_use=False):
    return np.array([pair_features(s1, s2, model, variant=variant, is_use=is_use) for s1, s2 in pairs])
