# RL-CVRP
Koda za mojo diplomo, odprtokodna rešitev

Za reševanje problema CVRP uporabim rekurenčno nevronsko mrežo GRU z mehanizmom pozornosti. Po fazi učenja, ki traja toliko časa kot uporaba stohastičnih optimizacijskih algoritmov na več tisoč primerih, dobimo na majhnih grafih  primerljivo dobre rezultate, na večjih grafih pa se zaradi povečevanja kompleksnosti problema nevronski model ne uči več dovolj hitro in je slabši. Med vložitvami grafov, ki sem jih preizkusil, dajeta najboljše rezultate node2vec in GraRep.

Začetna koda pobrana iz https://github.com/mveres01/pytorch-drl4vrp.

vrp.py - koda za generiranje VRP problemov, ocenjevalna funkcija, maskirna funkcija

model.py - koda za definicijo mojega modela, GRU enota, nivo pozornosti, glavni loop za napovedovanje zaporedja

trainer.py - koda za učenje mojega modela, glavni loop za treniranje, epohe, validacijo

genAlg_and_anneal.py - koda za uporabo genetskega algoritma ter simuliranega ohlajanja

Uporaba:
``python trainer.py`` za učenje nevronske mreže

``python genAlg_and_anneal.py`` za izvajanje genetskega algoritma, za izvajanje simuliranega ohlajanja v tej datoteki nastavi boolean GA na False

Za podrobnosti o parametrih preglej datoteko trainer.py
