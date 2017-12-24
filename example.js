var n = new neuralNet([9,10,10,3]);
var data = [
[[1,1,1,1,1,1,1,1,1], [0,0,1]], //[input], [desired]
[[1,1,1,1,1,1,1,1,0], [1,0,0]], //[input], [desired]
[[0,0,0,1,1,1,1,1,1], [0,1,0]]  //[input], [desired]
]

function mousePressed(){
  //run it with logging everything
  n.fullTrain(0.01, 10000, data, true);
  //or run it without logging.
  n.fullTrain(0.01, 10000, data);
}

function draw(){
  //This is needed or the mousePressed does not work!!!
}
