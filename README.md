# Image-Classification-on-Cifar10-with-Tensorflow-2
<p>This repository summaries some image classification architectures in computer vision that learners should know because most of them are fundamental and important to understand more complex architectures which use them as backbones. These models will be coded by Tensorflow 2 (2.6) via model sub-classing, the training and evaluating processes will be customed manually without high level API Keras.<br>
The dataset which is used for classification is CIFAR10 <br>
  (read more about CIFAR10 at here : https://www.cs.toronto.edu/~kriz/cifar.html).<br>
After training and evaluating , loss and accuracy histories will be plotted . A table benchmark will be shown to compare accuracy and number of parameter between these models.
</p>
<h3>Benchmark on CIFAR10</h3>
<table>
  <tr>
    <th>Model</th>
    <th>Test Accuracy</th>
    <th>Number of parameter</th>
  </tr>
  <tr>
    <td>VGG11</td>
    <td></td>
    <td>28m</td>
  </tr>
  <tr>
    <td>VGG13</td>
    <td></td>
    <td>28m</td>
  </tr>
  <tr>
    <td>VGG16</td>
    <td></td>
    <td>33m</td>
  </tr>
  <tr>
    <td>VGG19</td>
    <td></td>
    <td>38m</td>
  </tr>
  <tr>
    <td>ResNet18</td>
    <td></td>
    <td></td>
  </tr>
</table>
