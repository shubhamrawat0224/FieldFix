# Chunking Report

## Document counts by strategy

### Chunk size: 40 words
Document                          | Fixed | Recursive | Semantic
--------------------------------- | ----- | --------- | --------
Amperia Company Policy            | 10    | 12        | 14      
Amperia Customer Escalation Rules | 8     | 11        | 10      
Amperia Field Safety              | 9     | 12        | 12      
Amperia Parts Inventory Rules     | 8     | 11        | 11      
Amperia Refrigeration Guide       | 9     | 11        | 10      
Amperia Work Order Protocol       | 10    | 13        | 12      

### Chunk size: 200 words
Document                          | Fixed | Recursive | Semantic
--------------------------------- | ----- | --------- | --------
Amperia Company Policy            | 2     | 8         | 9       
Amperia Customer Escalation Rules | 2     | 6         | 5       
Amperia Field Safety              | 2     | 7         | 5       
Amperia Parts Inventory Rules     | 2     | 6         | 5       
Amperia Refrigeration Guide       | 2     | 6         | 4       
Amperia Work Order Protocol       | 2     | 7         | 2       

### Example: a small fixed chunk splits two related facts that recursive chunking keeps together

Document: `Amperia Refrigeration Guide`

The excerpt below contains a symptom cluster and a compressor diagnosis in the same operational paragraph.

`fixed-size (40 words, 10 overlap)` produced 3 chunks, while `recursive` produced 2 chunk because the section stays intact until the max-cap is exceeded.

Excerpt:
> A refrigeration case or walk-in unit that is not cooling may show one or more of the following: high head pressure, low evaporator airflow, warm product temperatures, repeated short cycling, or a fan motor that fails to start. Frost buildup on the evaporator can indicate restricted airflow, failed defrost control, or a refrigerant-related issue. A compressor that starts and then trips on overload is often a sign of overheating, failed capacitor, or a blocked condenser.

This matters because the technician needs the relationship between the cooling symptom and the compressor overload diagnosis in one retrieval unit. A short fixed-size boundary can separate these into different chunks even though they describe one root-cause chain.

