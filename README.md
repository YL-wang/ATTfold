# ATTfold
ATTfold method has 5 folders, we will introduce them separately.<br>
The following is the file distribution and introduction in each folder.<br>

#### Model construction and evaluation index establishment
>ATTfold
>>common <br>
>>>config.py<br>
>>>utils.py<br>

>>ATTfold_model.py <br>
>>data_generator.py <br>
>>evaluation.py <br>
>>README.md

#### Preprocessing of RNA sequence and structure data
>data_preprocess
>>rnastralign_512 <br>
>>>test_no_redundant_512.pickle

>>rnastralign_data.py <br>
>>test_de_redundancy.py <br>
>>README.md

#### ATTfold model training
>experiment_rnastralign
>>map <br>
>>ATTfold_learning.py <br>
>>config.json <br>
>>save_v.py <br>
>>README.md

#### Saved file of ATTfold model training parameters
>models_ckpt
>>ATTfold_0.809.pt <br>
>>README.md

#### Original data set
>raw_data
>>README.md
