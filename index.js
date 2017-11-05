var http = require('http');
const Encog = require('encog');
const XORdataset = Encog.Utils.Datasets.getXORDataSet();

// create a neural network

http.createServer(function (req, res) {
  res.writeHead(200, {'Content-Type': 'text/plain'});
  res.end('Hello Node.JS! '+ XORdataset.input   );
  const network = new Encog.Networks.Basic();
  network.addLayer(new Encog.Layers.Basic(null, true, 2));
  network.addLayer(new Encog.Layers.Basic(new Encog.ActivationFunctions.Sigmoid(), true, 4));
  network.addLayer(new Encog.Layers.Basic(new Encog.ActivationFunctions.Sigmoid(), false, 1));
  network.randomize();

  const train = new Encog.Training.Propagation.Back(network, XORdataset.input, XORdataset.output);

  Encog.Utils.Network.trainNetwork(train, {maxIterations: 250});
  const accuracy = Encog.Utils.Network.validateNetwork(network, XORdataset.input, XORdataset.output);
  console.log('Accuracy:', accuracy);

}).listen(8080);
console.log('Server running at http://localhost:8080/');
