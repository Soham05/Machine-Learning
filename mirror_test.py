import pandas as pd
import numpy as np
import ast
from sklearn.metrics.pairwise import cosine_similarity

def load_and_parse_data(plans_path: str, claims_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the CSVs, standardizes the join keys, and safely parses the stringified embeddings.
    """
    print("Loading datasets...")
    plans_df = pd.read_csv(plans_path)
    claims_df = pd.read_csv(claims_path)

    # 1. Standardize the keys to ensure exact matching (strip whitespace, uppercase)
    plans_df['plan_id'] = plans_df['plan_id'].astype(str).str.strip().str.upper()
    claims_df['xref_plan_code'] = claims_df['xref_plan_code'].astype(str).str.strip().str.upper()

    # 2. Safely parse the stringified embeddings back into actual Python lists/NumPy arrays
    print("Parsing embeddings from CSV strings to NumPy arrays...")
    # Using ast.literal_eval safely evaluates the string "[0.1, 0.2]" into a python list [0.1, 0.2]
    plans_df['plan_embedding'] = plans_df['plan_embedding'].apply(lambda x: np.array(ast.literal_eval(x)))
    claims_df['claims_semantic_embeddings'] = claims_df['claims_semantic_embeddings'].apply(lambda x: np.array(ast.literal_eval(x)))

    return plans_df, claims_df

def execute_mirror_test(plans_df: pd.DataFrame, claims_df: pd.DataFrame, top_k: int = 3):
    """
    Calculates cosine similarity between claims and plans, and reports Top-1 and Top-K accuracy.
    """
    print("\nExecuting Vector Similarity Search...")
    
    # 1. Create continuous 2D matrices for optimized math
    # Shape will be (300, 768) for plans and (10, 768) for claims
    plan_matrix = np.stack(plans_df['plan_embedding'].values)
    claim_matrix = np.stack(claims_df['claims_semantic_embeddings'].values)
    
    # Extract the plan IDs as a list so we can easily map the matrix indices back to the IDs
    plan_id_list = plans_df['plan_id'].tolist()

    # 2. Calculate the Cosine Similarity Matrix
    # Returns a matrix of shape (10, 300) where each row represents a claim's similarity to all 300 plans
    similarity_matrix = cosine_similarity(claim_matrix, plan_matrix)

    top_1_correct = 0
    top_k_correct = 0
    results = []

    # 3. Evaluate the matches
    for idx, row in claims_df.iterrows():
        true_plan_id = row['xref_plan_code']
        
        # Get the similarities for this specific claim
        similarities = similarity_matrix[idx]
        
        # Get the indices of the highest similarities (sorted descending)
        # argsort sorts ascending, so we use [::-1] to reverse it to descending
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Map those indices back to the actual Plan IDs and grab their similarity scores
        top_predicted_plans = [plan_id_list[i] for i in top_indices]
        top_predicted_scores = [similarities[i] for i in top_indices]

        # 4. Check for Accuracy
        is_top_1 = true_plan_id == top_predicted_plans[0]
        is_top_k = true_plan_id in top_predicted_plans
        
        if is_top_1:
            top_1_correct += 1
        if is_top_k:
            top_k_correct += 1

        # Store for the report
        results.append({
            'True_Plan_ID': true_plan_id,
            'Predicted_Top_1': top_predicted_plans[0],
            'Top_1_Score': round(top_predicted_scores[0], 4),
            'Predicted_Top_K': top_predicted_plans,
            'Top_1_Match': is_top_1,
            'Top_K_Match': is_top_k
        })

    # 5. Print the Analytics Report
    print("\n" + "="*50)
    print("MIRROR TEST RESULTS")
    print("="*50)
    
    results_df = pd.DataFrame(results)
    print(results_df[['True_Plan_ID', 'Predicted_Top_1', 'Top_1_Score', 'Top_1_Match', 'Top_K_Match']].to_string(index=False))
    
    print("\n" + "="*50)
    print("ACCURACY METRICS")
    print("="*50)
    total_claims = len(claims_df)
    print(f"Total Claims Cohorts Tested: {total_claims}")
    print(f"Total XML Plans in Knowledge Base: {len(plans_df)}")
    print(f"Top-1 Accuracy:  {(top_1_correct / total_claims) * 100:.2f}%")
    print(f"Top-{top_k} Accuracy:  {(top_k_correct / total_claims) * 100:.2f}%")
    
    return results_df

# --- Execution ---
if __name__ == "__main__":
    PLANS_CSV = "crx_embedded_plans.csv"
    CLAIMS_CSV = "crx_ibc_model.csv"
    
    # Run the pipeline
    plans_dataframe, claims_dataframe = load_and_parse_data(PLANS_CSV, CLAIMS_CSV)
    detailed_results = execute_mirror_test(plans_dataframe, claims_dataframe, top_k=3)




#### --------- Diagnostics ---------- ####
import pandas as pd
import numpy as np
import ast
from sklearn.metrics.pairwise import cosine_similarity

# Load data
plans_df = pd.read_csv("crx_embedded_plans.csv")
claims_df = pd.read_csv("crx_ibc_model.csv")

# Clean IDs
plans_df['plan_id'] = plans_df['plan_id'].astype(str).str.strip().str.upper()
claims_df['xref_plan_code'] = claims_df['xref_plan_code'].astype(str).str.strip().str.upper()

# 1. THE ID OVERLAP CHECK
xml_plans = set(plans_df['plan_id'].unique())
claim_plans = set(claims_df['xref_plan_code'].unique())
overlap = xml_plans.intersection(claim_plans)

print("="*50)
print(f"Total Unique XML Plans: {len(xml_plans)}")
print(f"Total Unique Claims Plans: {len(claim_plans)}")
print(f"ACTUAL OVERLAP (Plans in both files): {len(overlap)}")
print("="*50)

if len(overlap) == 0:
    print("\nFATAL ERROR: There are no matching Plan IDs between your two files.")
    print(f"Sample XML IDs: {list(xml_plans)[:3]}")
    print(f"Sample Claim IDs: {list(claim_plans)[:3]}")
else:
    # 2. THE SEMANTIC GAP CHECK
    test_id = list(overlap)[0]
    
    xml_text = plans_df[plans_df['plan_id'] == test_id]['semantic_text'].iloc[0]
    claim_text = claims_df[claims_df['xref_plan_code'] == test_id]['claims_semantic_text'].iloc[0]
    
    print(f"\n--- SIDE-BY-SIDE COMPARISON FOR PLAN: {test_id} ---")
    print("\n[TOWER 1: XML TEXT]")
    print(xml_text[:500] + "..." if len(xml_text) > 500 else xml_text)
    
    print("\n[TOWER 2: CLAIMS TEXT]")
    print(claim_text[:500] + "..." if len(claim_text) > 500 else claim_text)

    # 3. THE SCORE COLLAPSE CHECK
    plans_df['plan_embedding'] = plans_df['plan_embedding'].apply(lambda x: np.array(ast.literal_eval(x)))
    claims_df['claims_semantic_embeddings'] = claims_df['claims_semantic_embeddings'].apply(lambda x: np.array(ast.literal_eval(x)))
    
    plan_matrix = np.stack(plans_df['plan_embedding'].values)
    claim_vector = claims_df[claims_df['xref_plan_code'] == test_id]['claims_semantic_embeddings'].values[0].reshape(1, -1)
    
    similarities = cosine_similarity(claim_vector, plan_matrix)[0]
    print("\n[SIMILARITY SCORE DISTRIBUTION]")
    print(f"Highest Score to ANY plan: {np.max(similarities):.4f}")
    print(f"Lowest Score to ANY plan: {np.min(similarities):.4f}")
    
    # Find the score for the actual correct plan
    correct_plan_idx = plans_df.index[plans_df['plan_id'] == test_id].tolist()[0]
    print(f"Score for the CORRECT plan ({test_id}): {similarities[correct_plan_idx]:.4f}")


##### =----- Architectural Context -----
# PBM Semantic Matching Architecture (Two-Tower Model)


# PBM Semantic Vector Retrieval Architecture (Two-Tower Model)

## 1. Project Objective
Build an AI-driven vector recommendation engine that maps historical pharmacy claims behavior to their governing XML plan design rules. We are translating complex tabular database math into natural language, and using GCP Vertex AI (`text-embedding-005`) to match them in a 768-dimensional latent vector space using Cosine Similarity.

---

## 2. Tower 1: The Rule Space (Ground Truth)
**Status: COMPLETED & FROZEN**

**Objective:** Parse highly nested PBM Plan Design XML files into token-efficient, dense semantic paragraphs that an LLM embedding model can accurately map to vector space.
* **Input:** Raw XML files containing thousands of tags detailing plan rules (Deductibles, OOP Max, Tiered Copays, Days Supply, Utilization Management).
* **Engineering Logic:**
  * Uses recursive traversal to flatten the XML tree.
  * Employs "Sibling Grouping" to prevent token bloat (combining child nodes under a single parent path declaration).
  * Uses a "Breadcrumb" path format to strictly preserve the exact XML lineage without writing conversational filler.
* **Example Output Text:** `[cvsBenefitContainer > benefitClientAndPlanDetails > general] attributes: retail high dollar is 5000, home delivery high dollar is 5000. [cvsBenefitContainer > benefitVersionDetails > costShareNetwork] attributes: tier 1 copay is 10.0, tier 2 copay is 30.0.`
* **Output Artifact:** `crx_embedded_plans.csv` containing `plan_id`, `semantic_text`, and `plan_embedding`.

---

## 3. Tower 2: The Claims Space (Current Engineering Focus)
**Status: IN PROGRESS / DEBUGGING**

**Objective:** Process ~250 columns of raw, tabular pharmacy claims (both paid and rejected), group them by `FINAL_PLAN_CODE`, and calculate statistical aggregates. These aggregates must be serialized into a natural language "Semantic Fingerprint" that perfectly mimics the vocabulary of Tower 1.
* **Input:** Historical claims data where all columns are currently typed as `Char` (strings).
* **Engineering Logic:**
  * **Safe Casting:** Dynamically identify financial columns (`Amount`, `Cost`, `DED`, `OOP`) and safely cast them to `float`, coercing errors to `NaN`.
  * **Paid Claims Math:** Calculate the `max()` for continuous accumulators (Deductibles/OOP) and the `mode()` for categorical financials (Tiered Copays, Days Supply limits).
  * **Rejected Claims Math:** Calculate frequency distributions for rejection codes (e.g., Prior Auth, DAW penalties, Specialty restrictions).
  * **Graceful Omission (CRITICAL):** If an aggregation returns `NaN`, `None`, or an empty string, the metric MUST be completely omitted from the text generation. Do not output strings like "is nan" or "0.0".
* **The "Semantic Bridge" Rule:** The output text must strictly focus on declarative plan design math. NEVER include operational transaction noise (e.g., 'Claim Sequence', 'NPI_ID') or internal database keys (e.g., 'Pharmacy Price Schd Name is PCMX014662'). The embedding model cannot match internal DB keys to XML business rules.
* **Output Artifact:** `crx_ibc_model.csv` containing `xref_plan_code`, `claims_semantic_text`, and `claims_semantic_embeddings`.

---

## 4. The Mirror Test (Validation & Evaluation)
**Status: FAILING (Currently 0% Accuracy due to Semantic Collapse)**

**Objective:** The Mirror Test is our strict validation script. It proves whether the Tower 2 claims embeddings successfully map back to their exact Tower 1 XML embeddings. 
* **The Mechanism:** 1. Loads the 768-dimensional float arrays from both CSVs.
  2. Uses `sklearn.metrics.pairwise.cosine_similarity` to perform a matrix multiplication, comparing every claims vector against every XML vector.
  3. Sorts the similarity scores in descending order to find the Top-K closest matches.
* **Accuracy Metrics:**
  * **Top-1 Match:** The true `xref_plan_code` from Tower 2 is the absolute #1 highest cosine similarity score in Tower 1.
  * **Top-3 Match:** The true plan ID appears in the top 3 highest scores.
* **Why it is currently failing:** The textual vocabulary between the two towers is completely misaligned. Tower 1 speaks in business rules ("retail high dollar is 5000"). Tower 2 is currently outputting database noise ("Affiliation Code 00039 (32%)"). 
* **Goal for VS Code AI:** Refactor Tower 2's text generation to strip the noise and output clean, mathematically dense sentences so the Cosine Similarity Mirror Test achieves >80% Top-3 accuracy.
