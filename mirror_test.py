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
