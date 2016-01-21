# deepsea
Reimplementation of DeepSEA using TensorFlow.

To get everything set up:

  1. [Install TensorFlow][tf]
  1. Install other requirements, possibly in a virtualenv:
    `pip install -r requirements.txt`
  1. Download Jeff's [processed version][data] of the data. You'll need to put `train10k.mat` in the repo's root directory.
  
To train a model, run:

    ./deepsea.py

While this is training, you can inspect its state using [TensorBoard][tb]:

    tensorboard /tmp/deepsea_logs

See also:

  * [DeepSEA paper][paper] in Nature Methods, including the supplemental
  * [Jeff's Notes][jn]
  * [Dan's Notes][dn]

[tf]: https://www.tensorflow.org/versions/master/get_started/os_setup.html
[tb]: https://www.tensorflow.org/versions/master/how_tos/summaries_and_tensorboard/index.html
[data]: https://drive.google.com/file/d/0ByrU_xuJMu6YWS1rN3p2bmFabzg/view
[paper]: http://www.nature.com/nmeth/journal/v12/n10/full/nmeth.3547.html
[jn]: https://docs.google.com/document/d/1dp2Kn7258Tttd_-FEnM05HOUJzevq9B9GJn2lngcrfo/edit#heading=h.nh8yx5qspnzv
[dn]: https://docs.google.com/document/d/1-PrZB7IcNiADFU84a0H4AcUxpcPCq3PXev1SQPsMwXE/edit
