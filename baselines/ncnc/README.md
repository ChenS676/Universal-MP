Thisrepositorycontainstheofficialcodeforthepaper[NeuralCommonNeighborwithCompletionforLinkPrediction](https://arxiv.org/pdf/2302.00890.pdf).

**Environment**

TestedCombination:
torch1.13.0+pyg2.2.0+ogb1.3.5

```
condaenvcreate-fenv.yaml
```

**PrepareDatasets**

```
pythonogbdataset.py
```

**ReproduceResults**

Weimplementthefollowingmodels.

|name|$model|commandchange|
|----------|-----------|--------------------|
|GAE|cn0||
|NCN|cn1||
|NCNC|incn1cn1||
|NCNC2|incn1cn1|add--depth2--splitsize131072|
|GAE+CN|scn1||
|NCN2|cn1.5||
|NCN-diff|cn1res||
|NoTLR|cn1|delete--maskinput|

Toreproducetheresults,pleasemodifythefollowingcommandsasshowninthetableabove.

Cora
```
pythonNeighborOverlap.py--xdp0.7--tdp0.3--pt0.75--gnnedp0.0--preedp0.4--predp0.05--gnndp0.05--probscale4.3--proboffset2.8--alpha1.0--gnnlr0.0043--prelr0.0024--batch_size1152--ln--lnnn--predictor$model--datasetCora--epochs100--runs10--modelpuregcn--hiddim256--mplayers1--testbs8192--maskinput--jk--use_xlin--tailact
```

Citeseer
```
pythonNeighborOverlap.py--xdp0.4--tdp0.0--pt0.75--gnnedp0.0--preedp0.0--predp0.55--gnndp0.75--probscale6.5--proboffset4.4--alpha0.4--gnnlr0.0085--prelr0.0078--batch_size384--ln--lnnn--predictor$model--datasetCiteseer--epochs100--runs10--modelpuregcn--hiddim256--mplayers1--testbs4096--maskinput--jk--use_xlin--tailact--twolayerlin
```

Pubmed
```
pythonNeighborOverlap.py--xdp0.3--tdp0.0--pt0.5--gnnedp0.0--preedp0.0--predp0.05--gnndp0.1--probscale5.3--proboffset0.5--alpha0.3--gnnlr0.0097--prelr0.002--batch_size2048--ln--lnnn--predictor$model--datasetPubmed--epochs100--runs10--modelpuregcn--hiddim256--mplayers1--testbs8192--maskinput--jk--use_xlin--tailact
```

collab
```
pythonNeighborOverlap.py--xdp0.25--tdp0.05--pt0.1--gnnedp0.25--preedp0.0--predp0.3--gnndp0.1--probscale2.5--proboffset6.0--alpha1.05--gnnlr0.0082--prelr0.0037--batch_size65536--ln--lnnn--predictor$model--datasetcollab--epochs100--runs10--modelgcn--hiddim64--mplayers1--testbs131072--maskinput--use_valedges_as_input--res--use_xlin--tailact
```

ppa
```
pythonNeighborOverlap.py--xdp0.0--tdp0.0--gnnedp0.1--preedp0.0--predp0.1--gnndp0.0--gnnlr0.0013--prelr0.0013--batch_size16384--ln--lnnn--predictor$model--datasetppa--epochs25--runs10--modelgcn--hiddim64--mplayers3--maskinput--tailact--res--testbs65536--proboffset8.5--probscale4.0--pt0.1--alpha0.9--splitsize131072
```

ThefollowingdatasetsuseseparatecommandsforNCNandNCNC.Touseothermodels,pleasemodifyNCN'scommand.NotethatNCNCmodelsinthesedatasetsinitializeparameterswithtrainedNCNmodelstoacceleratetraining.Pleaseuseourpre-trainedmodelorrunNCNfirst.

citation2
```
pythonNeighborOverlapCitation2.py--xdp0.0--tdp0.3--gnnedp0.0--preedp0.0--predp0.2--gnndp0.2--gnnlr0.0088--prelr0.0058--batch_size32768--ln--lnnn--predictorcn1--datasetcitation2--epochs20--runs10--modelpuregcn--hiddim64--mplayers3--res--testbs65536--use_xlin--tailact--proboffset4.7--probscale7.0--pt0.3--trndeg128--tstdeg128--save_gemb


pythonNeighborOverlapCitation2.py--xdp0.0--tdp0.3--gnnedp0.0--preedp0.0--predp0.2--gnndp0.2--gnnlr0.0088--prelr0.001--batch_size24576--ln--lnnn--predictorincn1cn1--datasetcitation2--epochs20--runs10--modelnone--hiddim64--mplayers0--res--testbs65536--use_xlin--tailact--loadgemb/citation2_puregcn_cn1.pt--proboffset-0.3--probscale1.4--pt0.25--trndeg96--tstdeg96--loadgemb/citation2_puregcn_cn1.pt
```


ddi
```
pythonNeighborOverlap.py--xdp0.05--tdp0.0--gnnedp0.0--preedp0.0--predp0.6--gnndp0.4--gnnlr0.0021--prelr0.0018--batch_size24576--ln--lnnn--predictorcn1--datasetddi--epochs100--runs10--modelpuresum--hiddim224--mplayers1--testbs131072--use_xlin--twolayerlin--res--maskinput--savemod

pythonNeighborOverlap.py--xdp0.05--tdp0.0--gnnedp0.0--preedp0.0--predp0.6--gnndp0.4--gnnlr0.0000000--prelr0.0025--batch_size24576--ln--lnnn--predictorincn1cn1--datasetddi--proboffset3--probscale10--pt0.1--alpha0.5--epochs2--runs10--modelpuresum--hiddim224--mplayers1--testbs24576--splitsize262144--use_xlin--twolayerlin--res--maskinput--loadmod
```