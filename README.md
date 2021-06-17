**Erik Storm**

**s3715671**

The code used for my bachelor thesis: Named entity recognition for Frisian text using machine learning.
Including data and the results achieved.

# Scores

Testing LinearSVC

                      precision  recall    f1-score     support

              B-LOC      0.643     0.721     0.680       165
              B-PER      0.805     0.788     0.796       382
              B-ORG      0.468     0.408     0.436        71
             B-MISC      0.548     0.475     0.509       240
              I-ORG      0.216     0.205     0.211        39
              I-PER      0.691     0.713     0.702       157
              I-LOC      0.235     0.211     0.222        19
             I-MISC      0.426     0.186     0.259       140
      
         micro avg      0.645     0.588     0.615      1213
         macro avg      0.504     0.463     0.477      1213
      weighted avg      0.626     0.588     0.600      1213

Testing naive Bayes

                    precision    recall  f1-score   support

           B-LOC      0.474     0.782     0.590       165
           B-PER      0.556     0.843     0.670       382
           B-ORG      0.247     0.324     0.280        71
          B-MISC      0.251     0.608     0.356       240
           I-ORG      0.114     0.231     0.153        39
           I-PER      0.610     0.726     0.663       157
           I-LOC      0.059     0.053     0.056        19
          I-MISC      0.062     0.179     0.093       140

       micro avg      0.348     0.634     0.450      1213
       macro avg      0.297     0.468     0.358      1213
    weighted avg      0.395     0.634     0.480      1213

# Information

Hierbij de behaalde resultaten dusver. Ik heb op dit moment geprobeerd om een zo hoog mogelijke F1 score te bereiken en heb daarbij deze scores behaald.
Voor het ontwikkelen van het programma heb ik ook gekeken naar de runtime van het programma en de standaard deviatie van de f1 scores bij de cross-validation.
Wellicht is het beter om sommige features te verwijderen om zo de runtime en standaard deviatie te verbeteren.

Er zit ook een groot verschil in trainingsduur.

LinearSVC training tijd: 23s

NaiveBayes training tijd: 3s
