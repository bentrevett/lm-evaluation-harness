## Numerical Reasoning in Conversational Finance Question Answering (ConvFinQA)

### Summary

The [ConvFinQA](https://github.com/czyssrs/ConvFinQA) dataset contains annotated publicly available earnings reports of S&P 500 companies from 1999 to 2019, collected from the FinTabNet dataset. Each question has plain text context and one table in the HTML format and asks for a numerical result that can be derived from the text and table.

Apart from the direct FinQA setup, the implementation takes the "turn" version of the ConvFinQA and uses all the dialogue questions, extending the context by each turn (multi-turn setup, starting with a context size of 0).

### License
MIT License

### Citation
  ```
  @inproceedings{chen-etal-2021-finqa,
    title = "{F}in{QA}: A Dataset of Numerical Reasoning over Financial Data",
    author = "Chen, Zhiyu  and
      Chen, Wenhu  and
      Smiley, Charese  and
      Shah, Sameena  and
      Borova, Iana  and
      Langdon, Dylan  and
      Moussa, Reema  and
      Beane, Matt  and
      Huang, Ting-Hao  and
      Routledge, Bryan  and
      Wang, William Yang",
    editor = "Moens, Marie-Francine  and
      Huang, Xuanjing  and
      Specia, Lucia  and
      Yih, Scott Wen-tau",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.300",
    doi = "10.18653/v1/2021.emnlp-main.300",
    pages = "3697--3711",
}
  ```
  ```
  @inproceedings{chen-etal-2022-convfinqa,
    title = "{C}onv{F}in{QA}: Exploring the Chain of Numerical Reasoning in Conversational Finance Question Answering",
    author = "Chen, Zhiyu  and
      Li, Shiyang  and
      Smiley, Charese  and
      Ma, Zhiqiang  and
      Shah, Sameena  and
      Wang, William Yang",
    editor = "Goldberg, Yoav  and
      Kozareva, Zornitsa  and
      Zhang, Yue",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.421",
    doi = "10.18653/v1/2022.emnlp-main.421",
    pages = "6279--6292",
}
  ```

### Groups, Tags, and Tasks

#### Groups

* `group_name`: `Short description`

#### Tags

* `tag_name`: `Short description`

#### Tasks

* `task_name`: `1-sentence description of what this particular task does`
* `task_name2`: ...

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
