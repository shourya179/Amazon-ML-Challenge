import anyascii
import unicodedata
import re

INDIAN_GEO_MAP = {
    # Bengali
    'পশ্চিমবঙ্গ': 'west bengal', 'কলকাতা': 'kolkata', 'হাওড়া': 'howrah', 'শিলিगुড়ি': 'siliguri',
    # Hindi / Marathi
    'महाराष्ट्र': 'maharashtra', 'मध्य प्रदेश': 'madhya pradesh', 'उत्तर प्रदेश': 'uttar pradesh',
    'तमिलनाडु': 'tamil nadu', 'தமிழ்நாடு': 'tamil nadu', 'गुजरात': 'gujarat', 'ગુજરાત': 'gujarat',
    'कर्नाटक': 'karnataka', 'राजस्थान': 'rajasthan', 'आंध्र प्रदेश': 'andhra pradesh',
    'तेलंगाना': 'telangana', 'केरल': 'kerala', 'ओडिशा': 'odisha', 'उड़ीसा': 'odisha',
    'बिहार': 'bihar', 'पंजाब': 'punjab', 'हरियाणा': 'haryana', 'असम': 'assam',
    'दिल्ली': 'delhi', 'नई दिल्ली': 'new delhi', 'मुंबई': 'mumbai', 'बेंगलुरु': 'bengaluru',
    'हैदराबाद': 'hyderabad', 'पुणे': 'pune', 'भोपाल': 'bhopal', 'अहमदाबाद': 'ahmedabad'
}

TOKEN_NORMALIZATION_MAP = {
    # Common transliteration phonetic fixes to standard English loan words
    'adity': 'aditya',
    'proprtij': 'properties', 'proprti': 'property',
    'kmstrksms': 'constructions', 'kmstrkshn': 'construction', 'knstraksn': 'constructions',
    'marketimg': 'marketing', 'markettim': 'marketing',
    'teknolojij': 'technologies', 'teknoloji': 'technology', 'teknoljis': 'technologies', 'teknolji': 'technology',
    'solyushms': 'solutions', 'solyushn': 'solution', 'saliushan': 'solution', 'saliushans': 'solutions',
    'emtrpraaaijez': 'enterprises', 'emtrpraaij': 'enterprise', 'emtarapraij': 'enterprise', 'emtarapraijes': 'enterprises',
    'imddstrij': 'industries', 'imddstri': 'industry',
    'korporeshn': 'corporation',
    'praivet': 'private', 'praaivett': 'private', 'praibhet': 'private',
    'limittedd': 'limited', 'limitet': 'limited',
    'elelpi': 'llp',
    'piraivet': 'private', 'piraiveett': 'private',
    'picins': 'business', 'picinnns': 'business',
    'shkti': 'shakti', 'skti': 'shakti',
    'arbn': 'urban',
    'prodkts': 'products', 'proddktts': 'products',
    'kulopl': 'global', 'kulloopl': 'global',
    'hspitaliti': 'hospitality',
    'sarbhises': 'services', 'sarbhis': 'service'
}

ADDR_MAP = {
    'col': 'colony', 'clny': 'colony', 'colny': 'colony',
    'sec': 'sector', 'sect': 'sector',
    'soc': 'society', 'socty': 'society',
    'hsg': 'housing', 'hsng': 'housing',
    'ext': 'extension', 'extn': 'extension',
    'encl': 'enclave', 'enclv': 'enclave',
    'cmpd': 'compound', 'est': 'estate', 'indl': 'industrial',
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'av': 'avenue', 'blvd': 'boulevard',
    'ln': 'lane', 'dr': 'drive', 'ct': 'court', 'pl': 'place', 'sq': 'square',
    'cir': 'circle', 'cres': 'crescent', 'cl': 'close', 'rte': 'route',
    'hwy': 'highway', 'pkwy': 'parkway', 'expy': 'expressway', 'exp': 'expressway',
    'fl': 'floor', 'flr': 'floor', 'apt': 'apartment', 'apts': 'apartments',
    'ste': 'suite', 'bldg': 'building', 'bldng': 'building',
    'opp': 'opposite', 'oppo': 'opposite', 'nr': 'near', 'adj': 'adjacent',
    'bhnd': 'behind', 'b/h': 'behind',
    'dist': 'district', 'distt': 'district',
    'teh': 'tehsil', 'taluk': 'taluka', 'tal': 'taluka',
    'stn': 'station', 'nagar': 'nagar', 'marg': 'marg',
    'no': 'number', 'num': 'number',
    'shp': 'shop', 'plt': 'plot', 'blk': 'block',
    'mkt': 'market', 'rgcy': 'regency',
    'twr': 'tower', 'twrs': 'towers',
    'po': 'post office',
    'gf': 'ground floor', 'ff': 'first floor', 'sf': 'second floor', 'tf': 'third floor'
}

LEGAL_SUFFIXES = {
    'pvt ltd', 'private limited', 'pvt', 'ltd', 'limited', 'inc', 'incorporated',
    'corp', 'corporation', 'llc', 'llp', 'co', 'company', 'enterprises', 'enterprise',
    'holding', 'holdings', 'group', 'services', 'solutions', 'technologies', 'tech',
    'industries', 'international', 'intl', 'gmbh', 'sa', 'sarl', 'sas', 'bv', 'nv',
    'plc', 'spa', 'srl', 'sl', 'cia', 'assoc', 'associates'
}

def clean_enhanced(text, is_address=False):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = unicodedata.normalize('NFKC', text)
    for k, v in INDIAN_GEO_MAP.items():
        if k in text:
            text = text.replace(k, f" {v} ")
    if any(ord(c) >= 128 for c in text):
        text = anyascii.anyascii(text)
    text = text.lower()
    text = re.sub(r'&', ' and ', text)
    text = re.sub(r'[/\\_\-+,.:;!?\'"()\[\]{}|@#*^~`]', ' ', text)
    tokens = text.split()
    tokens = [TOKEN_NORMALIZATION_MAP.get(t, t) for t in tokens]
    if is_address:
        tokens = [ADDR_MAP.get(t, t) for t in tokens]
    return ' '.join(tokens)

def clean_name_and_core(name):
    cleaned = clean_enhanced(name, is_address=False)
    tokens = cleaned.split()
    core_tokens = list(tokens)
    while len(core_tokens) > 1:
        if len(core_tokens) >= 2 and f"{core_tokens[-2]} {core_tokens[-1]}" in LEGAL_SUFFIXES:
            core_tokens = core_tokens[:-2]
        elif core_tokens[-1] in LEGAL_SUFFIXES:
            core_tokens = core_tokens[:-1]
        else:
            break
    core = ' '.join(core_tokens) if core_tokens else cleaned
    return cleaned, core

raw = 'আলফা হসপিটালিটি প্রাইভেট লিমিটেড'
cleaned, core = clean_name_and_core(raw)
print('RAW:    ', raw)
print('CLEANED:', cleaned)
print('CORE:   ', core)
