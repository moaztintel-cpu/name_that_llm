# name_that_llm

## Dataset field glossary

The following terms describe the judge-related fields used in the dataset.

| Term | Description |
|---|---|
| **JudgeA, JudgeB, JudgeC** | The three judges, named A, B, and C for convenience. |
| **XYZ judge scores** | A three-digit score representing the individual scores from JudgeA, JudgeB, and JudgeC, in that order: `(x, y, z)`. For example, `099` means JudgeA scored 0, while JudgeB and JudgeC each scored 9. |
| **English score** | English-language confidence score. |
| **Similarity** | Sequence similarity score. |
| **Score** | Other scores related to the LLM-as-a-Judge evaluation, excluding the English score and similarity score. |
