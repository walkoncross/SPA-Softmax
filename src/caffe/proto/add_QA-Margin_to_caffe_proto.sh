cp caffe.proto caffe.proto.bk
cat QA-Margin_ip_layer.proto >> caffe.proto
sed -in '/optional PythonParameter python_param/ a \ \ optional QuasiAngularMarginInnerProductParameter quasi_angular_margin_inner_product_param = 148;' caffe.proto
