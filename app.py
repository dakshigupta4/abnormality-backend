import re
import pdfplumber
import logging
import numpy as np
import easyocr
from PIL import Image
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from thefuzz import process

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize OCR
reader = easyocr.Reader(['en'], gpu=False)

# ---------------------------------------------------------
# 1. MASTER KEYWORDS (The "Dictionary")
# ---------------------------------------------------------
TEST_MAPPING = {
    "TSH": ["Thyroid Stimulating Hormone", "TSH", "TSH Ultra", "T.S.H"],
    "Total T3": ["Total T3", "Triiodothyronine", "T3"],
    "Total T4": ["Total T4", "Thyroxine", "T4"],
    "Vitamin D": ["Vitamin D", "25-OH Vitamin D", "Total 25 OH Vitamin D"],
    "Vitamin B12": ["Vitamin B12", "Cyanocobalamin", "Vit B12"],
    "HbA1c": ["HbA1c", "Glycosylated Hemoglobin"],
    "Glucose Fasting": ["Fasting Blood Sugar", "FBS", "Glucose Fasting"],
    "Glucose PP": ["Post Prandial", "PPBS", "Glucose PP"],
    "Hemoglobin": ["Hemoglobin", "Hb", "Haemoglobin"],
    "PCV": ["PCV", "Packed Cell Volume", "Hematocrit", "HCT"],
    "RBC Count": ["RBC Count", "Red Blood Cell Count", "Total RBC"],
    "MCV": ["MCV"],
    "MCH": ["MCH"],
    "MCHC": ["MCHC"],
    "RDW": ["RDW", "R.D.W"],
    "TLC": ["TLC", "WBC", "Total Leucocyte Count", "White Blood Cell"],
    "Platelet Count": ["Platelet Count", "PLT", "Platelets"],
    "Neutrophils": ["Neutrophils", "Polymorphs"],
    "Lymphocytes": ["Lymphocytes"],
    "Monocytes": ["Monocytes"],
    "Eosinophils": ["Eosinophils"],
    "Basophils": ["Basophils"],
    "Urea": ["Urea", "Blood Urea"],
    "Creatinine": ["Creatinine", "Serum Creatinine"],
    "Uric Acid": ["Uric Acid"],
    "Cholesterol": ["Cholesterol", "Total Cholesterol"],
    "Triglycerides": ["Triglycerides"],
    "HDL": ["HDL Cholesterol", "H.D.L"],
    "LDL": ["LDL Cholesterol", "L.D.L"]
}

ALL_KEYWORDS = [alias for sublist in TEST_MAPPING.values() for alias in sublist]

# ---------------------------------------------------------
# 2. INTELLIGENT EXTRACTORS
# ---------------------------------------------------------
def extract_range(text):
    """ Finds range like '10-20', '<50', or explicit markers like '(Low)' """
    if not text: return None, None, None
    
    # Normalize text
    text = re.sub(r'\s+', ' ', text).strip()

    # --- FIX: ZIP CODE & ADDRESS GUARD ---
    # If the text contains common address patterns (like "Delhi-110002"), kill it immediately.
    if re.search(r'\b1100\d{2}\b', text): # Matches Delhi Zip codes 1100xx
        return None, None, None

    # Pattern A: Standard Numeric Range "10.5 - 20.5"
    # FIX: Restrict the "units" group (middle part) to avoid matching words like "Delhi" or "Road"
    # We only allow spaces or common unit chars like %, /, g, d, L, m
    dash_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:[%/a-zA-Z]{0,5}\s*)?[-–to]\s*(\d+(?:\.\d+)?)", text)
    
    if dash_match:
        min_v = float(dash_match.group(1))
        max_v = float(dash_match.group(2))
        
        # VALIDITY CHECK: Range logic
        # If the numbers are huge (like Zip codes > 100000) or identical, ignore them.
        if max_v > 50000: 
            return None, None, None 
            
        return min_v, max_v, dash_match.group(0)

    # Pattern B: Less than "< 5.0"
    less = re.search(r"(?:<|less than)\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if less:
        return 0.0, float(less.group(1)), less.group(0)

    # Pattern C: More than "> 5.0"
    more = re.search(r"(?:>|more than)\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if more:
        return float(more.group(1)), 999999.0, more.group(0)

    # Pattern D: Explicit "(Low)" or "(L)"
    if re.search(r"[\(\[]\s*(?:Low|L)\s*[\)\]]", text, re.IGNORECASE):
        return 999999.0, 999999.0, "(Low)"

    # Pattern E: Explicit "(High)" or "(H)"
    if re.search(r"[\(\[]\s*(?:High|H)\s*[\)\]]", text, re.IGNORECASE):
        return -999999.0, -999999.0, "(High)"

    return None, None, None

def extract_value(text_source, range_str):
    """ Finds the test result value, ignoring dates/IDs """
    if not text_source: return None
    
    clean = text_source
    if range_str: 
        clean = text_source.replace(range_str, "")
    
    clean = clean.replace(",", "")

    nums = re.findall(r"(\d+(?:\.\d+)?)", clean)
    
    valid_nums = []
    for n in nums:
        try:
            f = float(n)
            # IGNORE Years and Zip Codes
            if f < 2000 or (f > 2100 and f < 10000): 
                valid_nums.append(f)
        except: continue

    if not valid_nums: return None
    return valid_nums[0]

# ---------------------------------------------------------
# 3. MULTI-LINE PARSER (The "Brain")
# ---------------------------------------------------------
def parse_text_block(full_text):
    results = []
    lines = full_text.split("\n")
    
    # Remove empty/junk lines early
    lines = [l.strip() for l in lines if len(l.strip()) > 3]
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 1. Check for Metadata/Junk keywords
        ignore_terms = ["Test Name", "Result", "Unit", "Reference", "Page", "Date", "Time", "Remark", "Method"]
        if any(x.lower() in line.lower() for x in ignore_terms):
            i += 1
            continue

        # 2. Identify Test Name
        text_for_matching = re.sub(r'[\d\W_]+', ' ', line)
        keyword = None
        
        # Exact match
        for safe in ["Hb", "PCV", "TLC", "RBC", "MCV", "MCH", "MCHC", "RDW", "TSH"]:
             if re.search(r'\b' + re.escape(safe) + r'\b', line, re.IGNORECASE):
                 keyword = safe
                 break
        
        # Fuzzy match
        if not keyword:
            match = process.extractOne(text_for_matching, ALL_KEYWORDS, score_cutoff=85)
            if match: keyword = match[0]

        if keyword:
            try:
                std_name = next(k for k, v in TEST_MAPPING.items() if keyword in v)
            except:
                i += 1
                continue

            # 3. CONTEXT MERGING
            next_line = lines[i+1] if (i + 1) < len(lines) else ""
            context_block = line + " " + next_line
            
            # Extract
            min_r, max_r, range_txt = extract_range(context_block)
            val = extract_value(context_block, range_txt)

            # 4. STRICT VALIDATION
            if val is not None and min_r is not None:
                results.append({
                    "test_name": std_name,
                    "value": val,
                    "min": min_r,
                    "max": max_r,
                    "range": range_txt
                })
        
        i += 1 

    return results

# ---------------------------------------------------------
# 4. FILE PROCESSING
# ---------------------------------------------------------
def analyze_file(f):
    raw_text = ""
    filename = f.filename.lower()

    try:
        if filename.endswith(".pdf"):
            with pdfplumber.open(f) as pdf:
                for page in pdf.pages:
                    txt = page.extract_text()
                    if not txt: continue
                    if "no test results" in txt.lower(): continue
                    if not re.search(r'\d', txt): continue
                    
                    raw_text += "\n" + txt
                    tables = page.extract_tables()
                    for tb in tables:
                        for row in tb:
                            raw_str = " ".join([str(c) for c in row if c])
                            raw_text += "\n" + raw_str

        else:
            # Image Strategy
            img = Image.open(f).convert('RGB')
            ocr_list = reader.readtext(np.array(img), detail=0, paragraph=False)
            raw_text = "\n".join(ocr_list)

    except Exception as e:
        print(f"Error reading file: {e}")
        return []

    # --- FIX: GLOBAL BIOPSY / NARRATIVE REPORT FILTER ---
    # If this flag is tripped, we return NOTHING.
    lower_raw = raw_text.lower()
    biopsy_markers = [
        "biopsy", "histopathology", "specimen examined", 
        "microscopic examination", "impression:", "clinical history", 
        "department of pathology", "cytology"
    ]
    
    # Rule: If markers exist BUT no clear table structure (lots of text, few numbers)
    match_count = sum(1 for m in biopsy_markers if m in lower_raw)
    if match_count >= 1:
        # Heuristic: Lab reports have many numbers. Biopsies have few.
        # If digits are < 5% of characters, it's text.
        digit_count = sum(c.isdigit() for c in raw_text)
        if len(raw_text) > 0 and (digit_count / len(raw_text)) < 0.05:
            return []

    return parse_text_block(raw_text)

# ---------------------------------------------------------
# 5. API ROUTES
# ---------------------------------------------------------
@app.route('/analyze', methods=['POST'])
def analyze_route():
    if 'file' not in request.files: return {"error": "No file"}, 400
    f = request.files['file']

    all_data = analyze_file(f)

    abnormals = []
    seen_tests = set()

    for item in all_data:
        name = item['test_name']
        val = item['value']
        
        is_low = val < item['min']
        is_high = val > item['max']

        if is_low or is_high:
            if name not in seen_tests:
                item['status'] = "Low" if is_low else "High"
                abnormals.append(item)
                seen_tests.add(name)

    return jsonify({
        "count": len(abnormals),
        "results": abnormals
    })

@app.route('/')
def ui():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Abnormality Scanner</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: system-ui, sans-serif; background: #f4f4f5; padding: 20px; color: #18181b; }
            .container { max-width: 600px; margin: 0 auto; }
            .box { 
                background: white; border: 2px dashed #d4d4d8; padding: 40px; 
                text-align: center; border-radius: 12px; cursor: pointer; 
            }
            .box:hover { border-color: #2563eb; background: #eff6ff; }
            .card {
                background: white; padding: 16px; margin-top: 12px; border-radius: 8px;
                display: flex; justify-content: space-between; align-items: center;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-left: 5px solid #ccc;
            }
            .Low { border-left-color: #3b82f6; }
            .High { border-left-color: #ef4444; }
            .val { font-size: 20px; font-weight: bold; }
            .badge { 
                font-size: 12px; padding: 3px 8px; border-radius: 12px; 
                color: white; margin-left: 8px; vertical-align: middle;
            }
            .Low .badge { background: #3b82f6; }
            .High .badge { background: #ef4444; }
            #loading { display:none; text-align:center; margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2 style="text-align:center">🚨 Abnormality Scanner</h2>
            <div class="box" onclick="document.getElementById('f').click()">
                <div style="font-size:32px">📄</div>
                <div><b>Upload Report</b></div>
                <input type="file" id="f" hidden>
            </div>
            <div id="loading">Scanning Document...</div>
            <div id="output" style="margin-top: 20px;"></div>
        </div>
        <script>
            document.getElementById('f').addEventListener('change', async (e) => {
                const file = e.target.files[0];
                if(!file) return;
                const fd = new FormData(); fd.append('file', file);
                document.getElementById('loading').style.display = 'block';
                document.getElementById('output').innerHTML = '';
                try {
                    const res = await fetch('/analyze', {method:'POST', body:fd});
                    const data = await res.json();
                    if(data.count === 0) {
                        document.getElementById('output').innerHTML = 
                        '<div style="text-align:center; color:green; margin-top:20px">✅ No Abnormalities Found (All Normal)</div>';
                    } else {
                        let html = '';
                        data.results.forEach(item => {
                            html += `
                            <div class="card ${item.status}">
                                <div>
                                    <div style="font-weight:bold">${item.test_name}</div>
                                    <div style="font-size:12px; color:#71717a">Ref: ${item.range}</div>
                                </div>
                                <div class="val">
                                    ${item.value}
                                    <span class="badge">${item.status}</span>
                                </div>
                            </div>`;
                        });
                        document.getElementById('output').innerHTML = html;
                    }
                } catch(err) {
                    alert("Error processing file");
                } finally {
                    document.getElementById('loading').style.display = 'none';
                }
            });
        </script>
    </body>
    </html>
    """)

if __name__ == '__main__':
    app.run(debug=True, port=5000)



