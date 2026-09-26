# Recommendation System Review Rules

Before completing a code change:

1. Run the existing unit tests.
2. Verify that FAISS indices map to the correct products.
3. Check that previously reviewed products are excluded.
4. Validate the cold-start routing behavior.
5. Ensure the reranker only returns candidate product IDs.
6. Check that similarity scores are not described as probabilities.
7. Report untested behavior and potential failures.

Do not modify the recommendation algorithm
without explaining the change.