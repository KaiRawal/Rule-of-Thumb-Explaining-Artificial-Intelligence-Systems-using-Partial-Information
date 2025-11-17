


Start by cloning and running code from [https://github.com/akshajkumarv/LLMResumeBiasAnalysis](https://github.com/akshajkumarv/LLMResumeBiasAnalysis) to get the LLM API predictions we wish to explain with Rule of Thumb. We use a csv of resume summaries from the categories: `information-technology`, `accountant`, `aviation`, `construction`, `chef`, `advocate`, `teacher`, and `sales`; filtered using `GPT-4.1-nano` to select only IT workers. We provide these results already in [./Dataset/classification.csv](./Dataset/classification.csv). We then explain the behaviour of the zero shot LLM classifier without making additional API calls.

Starting repo structure:

```
SyntheticResumeFiltering/
├── README.md
├── RUN.sh
├── Code/
│   ├── prep_RoT_inputs.py
│   ├── rot_class.py
│   ├── rule_of_thumb.py
│   ├── run_RoT.py
│   ├── viz.py
│   └── word_clouds.py
└── LLMAnalysis/
    └── ...
```

`RUN.sh` provides end-to-end code to generate figures 3 and 14: `$ bash RUN.sh`.

