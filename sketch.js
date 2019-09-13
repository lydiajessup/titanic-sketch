/////////////////////////////////////////////////////
//                                                 //
//     Neural Network Training with the Titanic    //
//                                                 //
/////////////////////////////////////////////////////

// By: Lydia Jessup
// Date: August 21, 2019
// Description: Using TF.js to train an neural network to predict survival on the titanic
// Also:  This is heavily based off of examples by Dan Shiffman and the tf.js team



///////////////////////////////////////////////
// Import, normalize and transform data
///////////////////////////////////////////////

// this is a standard procedure to normalize values between 0 and 1
function normalize(value, min, max) {
  if (min === undefined || max === undefined) {
    return value;
  }
  return (value - min) / (max - min);
}



const TRAIN_DATA_PATH = './titanic_train.csv';
const TEST_DATA_PATH = './titanic_test.csv';
//or http://127.0.0.1:3000/?

//// Make Constants from training data ////

// have to put in all min and max values in order to normalize
// age and fare -- for class and sex will do one hot encoding
// I got these values from before when I was cleaning the data
const AGE_MIN = 0.0;
const AGE_MAX = 80.0;
const FARE_MIN = 0.0;
const FARE_MAX = 512.0;

const NUM_SURVIVED_CLASSES = 2;
const TRAINING_DATA_LENGTH = 916;
const TEST_DATA_LENGTH = 393;

let model;
let csvTransform;
let trainingData;
let trainingValidationData;
let testValidationData;

function setup(){
//// Convert rows from the CSV into features and labels ////

//import with csvdataset?
// filename = "/titanic.csv";
// train_dataset = tf.data.experimental.CsvDataset(filename);
// console.log(train_dataset);
// this isn't working

// Each feature field is normalized within training data constants
// xs is the features and xy is the label we are predicting
const csvTransform = ({xs, ys}) => {
      const values = [
        normalize(xs.age, AGE_MIN, AGE_MAX),
        normalize(xs.fare, FARE_MIN, FARE_MAX),
        xs.is_female //since this is already binary 0 and 1 we don't need to normalize
        //leaving out class for now
      ];
      return {xs: values, ys: ys.survived};
    }

    //make training data
const trainingData =
    tf.data.csv(TRAIN_DATA_PATH, {columnConfigs: {survived: {isLabel: true}}})
        .map(csvTransform)
        .shuffle(TRAINING_DATA_LENGTH)
        .batch(100);

// Make training validation data from training data
const trainingValidationData =
    tf.data.csv(TRAIN_DATA_PATH, {columnConfigs: {survived: {isLabel: true}}})
        .map(csvTransform)
        .batch(TRAINING_DATA_LENGTH);

// Make test vaidation data from test data
const testValidationData =
    tf.data.csv(TEST_DATA_PATH, {columnConfigs: {survived: {isLabel: true}}})
        .map(csvTransform)
        .batch(TEST_DATA_LENGTH);


///////////////////////////////////////////////
// Create model
///////////////////////////////////////////////


//using sequential model meaning sequential layers of neurons in our network
model = tf.sequential();
//now add layers
//input shape is 3 since we have
//we are using relu and softmax with are standard and more documentation can be found on tf.js
model.add(tf.layers.dense({units: 20, activation: 'relu', inputShape: [3]}));
model.add(tf.layers.dense({units: 10, activation: 'relu'}));
//this is the output layer with 2 output classes we defined before
model.add(tf.layers.dense({units: 2, activation: 'softmax'}));

//now compile!
//adam is standard
model.compile({
  optimizer: tf.train.adam(),
  loss: 'sparseCategoricalCrossentropy',
  metrics: ['accuracy']
});

//add run here
run(trainingData);
}


async function run(trainingData) {
  console.log(model.summary());
  console.log(trainingData);

  await model.fitDataset(trainingData, {
    epochs: 20,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(`Epoch: ${epoch} - loss: ${logs.loss.toFixed(3)}`);
      }
    }
});
}


//another version?
// async function run() {
//   //let's check what we're putting into the model
//   console.log(model.summary());
//
//   await model.fit(trainingData.xs, trainingData.ys, {
//     shuffle: true,
//     validationSplit: 0.1,
//     epochs: 20,
//     callbacks: {
//         onEpochEnd: (epoch, logs) => {
//         console.log(epoch);
//         //lossP.html('loss: ' + logs.loss.toFixed(5));
//       },
//       onBatchEnd: async(batch, logs) => {
//         await tf.nextFrame();
//       },
//       onTrainEnd: () => {
//         console.log('finished')
//       },
//     },
//   });
// }
