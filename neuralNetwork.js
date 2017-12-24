class neuralNet {
  constructor(sizes){
    this.learnRate = 1; // A number to adjust the rate of adjustments TODO: further research on how this, or specific LR for every layers effect efficiency.
    this.bias = new Array(3);//Create an array of biases, one place for every layer. An array so we safe space in the console.log. May be less efficient in printing.
    this.bias.fill(1); //Make sure we put a one in every spot.
    this.nodeConfig = sizes; //Just so we can see the 'layout' of the network in the console.
    this.output = new Array(sizes[3]);
    this.a = new Array(sizes[0]); //A = input array. All the names are given according to the drawing I improvised... Horrible idea.
    this.w1 = new Array(sizes[1]);
    this.p = new Array(sizes[1]); //here we will store Wx * Iy or the other way around with the x and y.
    for(let i = 0; i < sizes[1]; i++){//Create array of weights with w1[size of hidden][size of inputs]
      var tempArray = new Array();
      for(let j = 0; j < sizes[0]; j++){
        tempArray.push(Math.random()*2 -1);
      }
      this.w1[i] = tempArray;
      this.p[i] = new Array(sizes[0]);
    }
    //Now we go make all the second layer.
    this.b = new Array(sizes[1]); //The outputs with the sigmoid applied of the first hidden layer.
    this.w2 = new Array(sizes[2]); //Second layer of weights.
    this.q = new Array(sizes[2]); //all the outputs from b multiplied by their corresponding weights.
    for(let i = 0; i < sizes[2]; i++){//Create array of weights with w1[size of hidden][size of inputs]
      var tempArray = new Array();
      for(let j = 0; j < sizes[1]; j++){
        tempArray.push(Math.random()*2 -1);
      }
      this.w2[i] = tempArray;
      this.q[i] = new Array(sizes[1]);
    }     //And the last hidden layer is here below.
    this.c = new Array(sizes[2]);
    this.w3 = new Array(sizes[3]);
    this.r = new Array(sizes[3]);
    this.dr = new Array(sizes[3]); //We will use dr as Derivatives of R. We use this as a groupname to simplify computation.
    for(let i = 0; i < sizes[3]; i++){//Create array of weights with w1[size of hidden][size of inputs]
      var tempArray = new Array();
      for(let j = 0; j < sizes[2]; j++){
        tempArray.push(Math.random()*2 -1);
      }
      this.w3[i] = tempArray;
      this.dr[i] = new Array(sizes[2]);
      this.dr[i].fill(1);
      this.r[i] = new Array(sizes[2]);
    }
    this.z = new Array(sizes[3]);//The output layer.
    var tempInput = new Array(sizes[0]);
    tempInput.fill(0);
    this.activate(tempInput); //Make sure we fill it with values before we can use it.
  }
  activate(input){
      this.a = input;
      var sum = 0;
      for(let i = 0; i < this.b.length; i++){
        for(let j = 0; j < this.a.length; j++){
          this.p[i][j] = this.w1[i][j] * this.a[j];
          sum += this.p[i][j];
        }
        this.b[i] = sigmoid(sum + this.bias[0]);
        sum = 0;
      }
      for(let i = 0; i < this.c.length; i++){
        for(let j = 0; j < this.b.length; j++){
          this.q[i][j] = this.b[j] * this.w2[i][j];
          sum += this.q[i][j];
        }
        this.c[i] = sigmoid(sum + this.bias[1]);
        sum = 0;
      }
      for(let i = 0; i < this.z.length; i++){
        for(let j = 0; j < this.c.length; j++){
          this.r[i][j] = this.c[j] * this.w3[i][j];
          sum += this.r[i][j];
        }
        this.z[i] = sigmoid(sum + this.bias[2]);
        sum = 0;
      }
      return this.z;
  }
  //////////////////////////////////////////////////////////////////////////////
  train(input, desired){
    this.activate(input);
    this.errorZ = new Array(this.z.length);
    for(let i = 0; i < this.z.length; i++){
      this.errorZ[i] = desired[i] - this.z[i];
    }
    this.adjustW1();
    this.adjustB1();
    this.activate(input);
    for(let i = 0; i < this.z.length; i++){
      this.errorZ[i] = desired[i] - this.z[i];
    }
    this.adjustW2();
    this.adjustB2();
    this.activate(input);
    for(let i = 0; i < this.z.length; i++){
      this.errorZ[i] = desired[i] - this.z[i];
    }
    this.adjustW3();
    this.adjustB3();
  }
  //////////////////////////////////////////////////////////////////////////////
  run(input){
    this.activate(input);
    for(let i = 0; i < this.z.length; i++){
      this.output[i] = Math.round(this.z[i]);
    }
    console.log(this.output);
  }
  //////////////////////////////////////////////////////////////////////////////
  adjustB1(){
    for(let i = 0; i < this.dr.length; i++){
      for(let j = 0; j < this.dr[i].length; j++){
        this.dr[i][j] = this.errorZ[i] * sigmoidD(this.z[i]) * this.w3[i][j] * sigmoidD(this.c[j]);//There used to be errorZ - z. However i deleted this because it was not correct.
      } //If there ever comes trouble: TODO: change it back ??? TODO: find out if that was a mistake or not. i dont have time for now.
    }
    var sum = 0;
    for(let j = 0; j < this.b.length; j++){
      for(let k = 0; k < this.c.length; k++){
        for(let l = 0; l < this.z.length; l++){
          sum += sigmoidD(this.b[j]) * this.w2[k][j] * this.dr[l][k];
        }
      }
    }
    this.bias[0] += sum;
  }
  adjustB2(){
    var sum = 0;
    for(let i = 0; i < this.w2.length; i++){
        for(let k = 0; k < this.z.length; k++){
          sum += this.errorZ[k] * sigmoidD(this.z[k]) * this.w3[k][i] * sigmoidD(this.c[i]) * 0.01;
        }
        this.bias[1] += sum;
        sum = 0;
      }
  }
  adjustB3(){
    var sum = 0; //Because we need to do a lot of adding. One entire layer shares the same bias.
    for(let i = 0; i < this.w3.length; i++){ //Loop through all the weights in the specified layer.
      sum += sigmoidD(this.z[i]) * this.errorZ[i];
    }
    this.bias[2] += sum;
    sum = 0;
  }
  adjustW1(){
    this.w1GlobalChange = 0;
    var count = 0;
    for(let i = 0; i < this.dr.length; i++){
      for(let j = 0; j < this.dr[i].length; j++){
        this.dr[i][j] = this.errorZ[i] * sigmoidD(this.z[i]) * this.w3[i][j] * sigmoidD(this.c[j]);//There used to be errorZ - z. However i deleted this because it was not correct.
      } //If there ever comes trouble: TODO: change it back ??? TODO: find out if that was a mistake or not. i dont have time for now.
    }
    var sum = 0;
    for(let i = 0; i < this.a.length; i++){ //For every
      for(let j = 0; j < this.b.length; j++){
        for(let k = 0; k < this.c.length; k++){
          for(let l = 0; l < this.z.length; l++){
            sum += this.a[i] * sigmoidD(this.b[j]) * this.w2[k][j] * this.dr[l][k];
          }
        }
        this.w1[j][i] += sum * this.learnRate;
        this.w1GlobalChange += Math.abs(sum * this.learnRate);
        count++;
        sum = 0;
      }
    }
    this.w1GlobalChange /= count;
  }

  adjustW2(){
    var sum = 0;
    var firstblock = 0;
    var count = 0;
    this.w2GlobalChange = 0;
    var count = 0;
    for(let i = 0; i < this.w2.length; i++){
      for(let j = 0; j < this.w2[i].length; j++){
        firstblock = this.b[j]* sigmoid(this.c[i]);
        for(let k = 0; k < this.z.length; k++){
          sum += firstblock * (this.errorZ[k] * sigmoidD(this.z[k]) * this.w3[k][i]);
        }
        this.w2[i][j] += sum * this.learnRate;
        this.w2GlobalChange += Math.abs(sum * this.learnRate);
        count++;
        sum = 0;
        firstblock = 0;
      }
    }
  this.w2GlobalChange /= count;
  }
  adjustW3(){
    this.w3GlobalChange = 0;
    var count = 0;
    for(let i = 0; i < this.w3.length; i++){
      for(let j = 0; j < this.w3[i].length; j++){
        this.w3[i][j] += this.errorZ[i] * this.c[j] * sigmoidD(this.z[i]) * this.learnRate;
        this.w3GlobalChange += Math.abs(this.errorZ[i] * this.c[j] * sigmoidD(this.z[i]) * this.learnRate * 0.1);
        count++;
      }
    }
    this.w3GlobalChange /= count;
  }
  ///////////////////////////////////////////////////////////////
  fullTrain(errorMargin, countMargin, data, log = false){
    var avrError = 1, avrCount = 0, e = 0;
    while((avrError > errorMargin) && (e < countMargin)){
      avrError = 0;
      for(let i = 0; i < data.length; i++){
        this.train(data[i][0],data[i][1]);
        for(let a = 0; a < this.errorZ.length; a++){
          avrError += Math.abs(this.errorZ[a]);
          if(log){
          console.log("abs = ", Math.abs(n.errorZ[a]).toFixed(5));
          }
          avrCount++;
        }
      }
      avrError /= avrCount;
      if(log){
      console.log(avrError)
      console.log("runtime = ", e);
      }
      avrCount = 0;
      e++;
      if(e == 10000) console.log("man this network fucked up.");
    }
    for(let i = 0; i < data.length; i++){
      this.run(data[i][0]);
    }
  }
}
//All the functions used. Don't mind those, just 'simple' maths.
function sigmoid(t) {
    return 1/(1+Math.pow(Math.E, -t)); // Go from any number to number from 0 to 1. sigmoid(0) = 0.5
}
function sigmoidD(t){
  return sigmoid(t) * (1-sigmoid(t)); //Sigmoid derivative
}
function rsigmoid(y){
  return log(y/(1-y)) //Reverse the sigmoid. Go from number between 0 and 1 to normal number.
}
