// from https://medium.freecodecamp.org/how-to-create-a-neural-network-in-javascript-in-only-30-lines-of-code-343dafc50d49
const { Layer, Network } = require('synaptic');

const inputLayer = new Layer(2);
const hiddenLayer = new Layer(3);
const outputLayer = new Layer(1);

inputLayer.project(hiddenLayer);
hiddenLayer.project(outputLayer);

const myNetwork = new Network({
    input: inputLayer,
    hidden: [hiddenLayer],
    output: outputLayer,
});

const learningRate = 0.3;

for (var i = 0; i < 20000; i++) {
    // 0,0 => 0
    myNetwork.activate([0, 0]);
    myNetwork.propagate(learningRate, [0]);
    // 0,1 => 1
    myNetwork.activate([0, 1]);
    myNetwork.propagate(learningRate, [1]);
    // 1,0 => 1
    myNetwork.activate([1, 0]);
    myNetwork.propagate(learningRate, [1]);
    // 1,1 => 0
    myNetwork.activate([1, 1]);
    myNetwork.propagate(learningRate, [0]);
}

function testNetwork(num = 1) {
    if (num <= 0) return;

    console.log(myNetwork.activate([0, 0]));
    // -> [0.015020775950893527]
    console.log(myNetwork.activate([0, 1]));
    // ->[0.9815816381088985]
    console.log(myNetwork.activate([1, 0]));
    // -> [0.9871822457132193]
    console.log(myNetwork.activate([1, 1]));
    // -> [0.012950087641929467]

    console.log('\n');
    testNetwork(num - 1);
}

testNetwork(5);
