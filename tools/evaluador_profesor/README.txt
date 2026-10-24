IR Evaluation Sample Files

The number of relevant documents may vary across the 15 queries.
Counts range from 10 to 15, matching the assignment specification. 

qrels_sample.tsv: query_id, document_id, relevance
rankings_sample.tsv: query_id, document_id, rank, score

Relevance: 1=partially relevant, 2=highly relevant. Documents absent from qrels
are treated as relevance 0 for this exercise.

Run:
python evaluate_ir_metrics.py rankings_sample.tsv qrels_sample.tsv
