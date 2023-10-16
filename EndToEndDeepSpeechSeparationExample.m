%% End-to-End Deep Speech Separation
% This example showcases an end-to-end deep learning network for speaker-independent 
% speech separation.
%% Introduction
% Speech separation is a challenging and critical speech processing task. A 
% number of speech separation methods based on deep learning have been proposed 
% recently, most of which rely on time-frequency transformations of the time-domain 
% audio mixture (See <docid:audio_ug#mw_40c4cb97-8ca7-4bc8-8ca0-f7eee39ca3c9 Cocktail 
% Party Source Separation Using Deep Learning Networks> for an implementation 
% of such a deep learning system). 
% 
% Solutions based on time-frequency methods suffer from two main drawbacks:
%% 
% * The conversion of the time-frequency representations back to the time domain 
% requires phase estimation, which introduces errors and leads to imperfect reconstruction. 
% * Relatively long windows are required to yield high resolution frequency 
% representations, which leads to high computational complexity and unacceptable 
% latency for real-time scenarios.
%% 
% In this example, you explore a deep learning speech separation network (based 
% on [1]) which acts directly on the audio signal and bypasses the issues arising 
% from time-frequency transformations.
%% Separate Speech using the Pretrained Network
% Download the Pretrained Network
% Before training the deep learning network from scratch, you will use a pretrained 
% version of the network to separate two speakers from an example mixture signal.
% 
% First, download the pretrained network and example audio files.
mytempdir='E:\speech_separation';
url = 'http://ssd.mathworks.com/supportfiles/audio/speechSeparation.zip';
downloadNetFolder = mytempdir;
netFolder = fullfile(downloadNetFolder,'speechSeparation');

if ~exist(netFolder,'dir')
    disp('Downloading pretrained network and audio files ...')
    unzip(url,downloadNetFolder)
end
% Prepare Test Signal
% Load two audio signals corresponding to two different speakers. Both signals 
% are sampled at 8 kHz.

Fs = 8000;
s1 = audioread(fullfile(netFolder,'speaker1.wav'));
s2 = audioread(fullfile(netFolder,'speaker2.wav'));
%% 
% Normalize the signals.

s1 = s1/max(abs(s1));
s2 = s2/max(abs(s2));
%% 
% Listen to a few seconds of each signal.

T = 5;
sound(s1(1:T*Fs))
pause(T)
%%
sound(s2(1:T*Fs))
pause(T)
%% 
% Combine the two signals into a mixture signal.

mix = s1+s2;
mix = mix/max(abs(mix));
%% 
% Listen to the first few seconds of the mixture signal.

sound(mix(1:T*Fs))
pause(T)
% Separate Speakers
% Load the parameters of the pretrained speech separation network.

load(fullfile(netFolder,'paramsBest.mat'),'learnables','states')
%% 
% Separate the two speakers in the mixture signals by calling the |separateSpeakers| 
% function.

[z1,z2] = separateSpeakers(mix,learnables,states,false);
%% 
% Listen to the first few seconds of the first estimated speech signal.

sound(z1(1:T*Fs))
pause(T)
%% 
% Listen to the second estimated signal.

sound(z2(1:T*Fs))
pause(T)
%% 
% To illustrate the effect of speech separation, plot the estimated and original 
% separated signals along with the mixture signal.

s1 = s1(1:length(z1));
s2 = s2(1:length(z2));
mix = mix(1:length(s1));

t  = (0:length(s1)-1)/Fs;

figure;
subplot(311)
plot(t,s1)
hold on
plot(t,z1)
grid on
legend('Speaker 1 - Actual','Speaker 1 - Estimated')
subplot(312)
plot(t,s2)
hold on
plot(t,z2)
grid on
legend('Speaker 2 - Actual','Speaker 2 - Estimated')
subplot(313)
plot(t,mix)
grid on
legend('Mixture')
xlabel('Time (s)')
% Compare to a Time-Frequency Transformation Deep Learning Network
% Next, you compare the performance of the network to the network developed 
% in the <docid:audio_ug#mw_40c4cb97-8ca7-4bc8-8ca0-f7eee39ca3c9 Cocktail Party 
% Source Separation Using Deep Learning Networks> example. This speech separation 
% network is based on traditional time-frequency representations of the audio 
% mixture (using the short-time Fourier transform, STFT, and the inverse short-time 
% Fourier transform, ISTFT).
%% 
% Download the pretrained network.

url = 'http://ssd.mathworks.com/supportfiles/audio/CocktailPartySourceSeparation.zip';

downloadNetFolder = mytempdir;
cocktailNetFolder = fullfile(downloadNetFolder,'CocktailPartySourceSeparation');

if ~exist(cocktailNetFolder,'dir')
    disp('Downloading pretrained network and audio files (5 files - 24.5 MB) ...')
    unzip(url,downloadNetFolder)
end
%% 
% The function |separateSpeakersTimeFrequency| encapsulates the steps required 
% to separate speech using this network|.| The function performs the following 
% steps:
%% 
% * Compute the magnitude STFT of the input time-domain mixture.
% * Compute a soft time-frequency mask by passing the STFT to the network.
% * Compute the STFT of the separated signals by multiplying the mixture STFT 
% by the mask.
% * Reconstruct the time-domain separated signals using ISTFT. The phase of 
% the mixture STFT is used.
%% 
% Refer to the <docid:audio_ug#mw_40c4cb97-8ca7-4bc8-8ca0-f7eee39ca3c9 Cocktail 
% Party Source Separation Using Deep Learning Networks> example for more details 
% about this network.
%% 
% Separate the two speakers.

[y1,y2] = separateSpeakersTimeFrequency(mix,cocktailNetFolder);
%% 
% Listen to the first separated signal.

sound(y1(1:Fs*T))
pause(T)
%% 
% Listen to the second separated signal.

sound(y2(1:Fs*T))
pause(T)
% Evaluate Network Performance using SI-SNR
% You will compare the two networks using the scale-invariant source-to-noise 
% ratio (SI-SNR) objective measure [1]. 
%% 
% Compute the SISNR for the first speaker with the end-to-end network.
% 
% First, normalize the actual and estimated signals.

s10 = s1 - mean(s1);
z10 = z1 - mean(z1);
%% 
% Compute the "signal" component of the SNR.

t = sum(s10.*z10) .* z10 ./ (sum(z10.^2)+eps);
%% 
% Compute the "noise" component of the SNR.

n = s1 - t;
%% 
% Now compute the SI-SNR (in dB).

v1 = 20*log((sqrt(sum(t.^2))+eps)./sqrt((sum(n.^2))+eps))/log(10);
fprintf('End-to-end network - Speaker 1 SISNR: %f dB\n',v1)
%% 
% The SI-SNR computation steps are encapsulated in the function |SISNR|. Use 
% the function to compute the SI-SNR of the second speaker with the end-to-end 
% network.

v2 = SISNR(z2,s2);
fprintf('End-to-end network - Speaker 2 SISNR: %f dB\n',v2)
%% 
% Next, compute the SI-SNR for each speaker for the STFT-based network.

w1 = SISNR(y1,s1(1:length(y1)));
w2 = SISNR(y2,s2(1:length(y2)));
fprintf('STFT network - Speaker 1 SISNR: %f dB\n',w1)
fprintf('STFT network - Speaker 2 SISNR: %f dB\n',w2)
%% Training the Speech Separation Network
% Examine the Network Architecture
% 
% 
% The network is based on [1] and consists of three stages: Encoding, mask estimation 
% or separation, and decoding.
%% 
% * The encoder transforms the time-domain input mixture signals into an intermediate 
% representation using convolutional layers.
% * The mask estimator computes one mask per speaker. The intermediate representation 
% of each speaker is obtained by multiplying the encoder's output by its respective 
% mask. The mask estimator is comprised of 32 blocks of convolutional and normalization 
% layers with skip connections between blocks.
% * The decoder transforms the intermediate representations to time-domain separated 
% speech signals using transposed convolutional layers.
%% 
% The operation of the network is encapsulated in |separateSpeakers|.
% Optionally Reduce the Dataset Size
% To train the network with the entire dataset and achieve the highest possible 
% accuracy, set |reduceDataset| to false. To run this example quickly, set |reduceDataset| 
% to true. This will run the rest of the example on only a handful of files.

reduceDataset = false;
%% 
% 
% Download the Training Dataset
% You use a subset of the LibriSpeech Dataset [2] to train the network. The 
% LibriSpeech Dataset is a large corpus of read English speech sampled at 16 kHz. 
% The data is derived from audiobooks read from the LibriVox project.
% 
% Download the LibriSpeech dataset. If |reduceDataset| is true, this steo is 
% skipped.

downloadDatasetFolder = mytempdir;
datasetFolder = fullfile(downloadDatasetFolder,"LibriSpeech","train-clean-100");
%{
if ~reduceDataset    
    filename = "train-clean-360.tar.gz";
    url = "http://www.openSLR.org/resources/12/" + filename;
    if ~isfolder(datasetFolder)
        gunzip(url,downloadDatasetFolder);
        unzippedFile = fullfile(downloadDatasetFolder,filename);
        untar(unzippedFile{1}(1:end-3),downloadDatasetFolder);
    end
end
%}
% Preprocess the Dataset
% The LibriSpeech dataset is comprised of a large number of audio files with 
% a single speaker. It does not contain mixture signals where 2 or more persons 
% are speaking simultaneously. 
% 
% You will process the original dataset to create a new dataset that is suitable 
% for training the speech separation network.
% 
% The steps for creating the training dataset are encapsulated in |createTrainingDataset|. 
% The function creates mixture signals comprised of utterances of two random speakers. 
% The function returns three audio datastores:
%% 
% * |mixDatastore| points to mixture files (where two speakers are talking simultaneously).
% * |speaker1Datastore| points to files containing the isolated speech of the 
% first speaker in the mixture.
% * |speaker2Datastore| points to files containing the isolated speech of the 
% second speaker in the mixture.

miniBatchSize = 4;
[mixDatastore,speaker1Datastore,speaker2Datastore] = createTrainingDataset(netFolder,datasetFolder,downloadDatasetFolder,reduceDataset,miniBatchSize);
%% 
% Combine the datastores. This ensures that the files stay in the correct order 
% when you shuffle them at the start of each new epoch in the training loop.

ds = combine(mixDatastore,speaker1Datastore,speaker2Datastore);
%% 
% Create a minibatch queue from the datastore.

mqueue = minibatchqueue(ds,'MiniBatchSize',miniBatchSize,'OutputEnvironment','cpu','OutputAsDlarray',false);
% Specify Training Options
% Define training parameters.
% 
% Train for 10 epochs.

if reduceDataset
    numEpochs = 1;
else
    numEpochs = 10; %#ok
end
%% 
% Specify the options for Adam optimization. Set the initial learning rate to 
% 1e-3. Use a gradient decay factor of 0.9 and a squared gradient decay factor 
% of 0.999.

learnRate = 1e-3;
averageGrad = [];
averageSqGrad = [];

gradDecay = 0.9;
sqGradDecay = 0.999;
%% 
% Train on a GPU if one is available. Using a GPU requires Parallel Computing 
% Toolbox™.

executionEnvironment = "auto"; % Change to "gpu" to train on a GPU.

duration = 4 * 8000;
% Set Up Validation Data
% You will use the test signal you previously employed to test the pretrained 
% network to compute a validation SI-SNR periodically during training.
% 
% If a GPU is available, move the validation signal to the GPU.

mix = dlarray(mix,'SCB');
if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    mix = gpuArray(mix);
end
%% 
% Define the number of iterations between validation SI-SNR computations.

numIterPerValidation = 50;
%% 
% Define a vector to hold the validation SI-SNR from each iteration.

valSNR = [];
%% 
% Define a variable to hold the best validation SI-SNR.

bestSNR = -Inf;
%% 
% Define a variable to hold the epoch in which the best validation score occurred.

bestEpoch = 1;
% Initialize Network
% Initialize the network parameters. |learnables| is a structure containing 
% the learnable parameters from the network layers. |states| is a structure containing 
% the states from the normalization layers.

[learnables,states] = initializeNetworkParams;
% Train the Network
% Execute the training loop. This can take many hours to run.
% 
% Note that there is no a priori way to associate the estimated output speaker 
% signals with the expected speaker signals. This is resolved by using Utterance-level 
% permutation invariant training (uPIT) [1]. The loss is based on computing the 
% SI-SNR. uPIT minimizes the loss over all permutations between outputs and targets. 
% It is defined in the function |uPIT|.
% 
% The validation SI-SNR is computed periodically. If the SI-SNR is the best 
% value to-date, the network parameters are saved to |params.mat|.

iteration = 0;

% Loop over epochs.
for jj =1:numEpochs

    % Shuffle the data
    shuffle(mqueue);

    while hasdata(mqueue)

        % Compute validation loss/SNR periodically
        if mod(iteration,numIterPerValidation)==0
            
            [z1,z2] = separateSpeakers(mix, learnables,states,false);
            
            l = uPIT(z1,s1,z2,s2);
            valSNR(end+1) = l; %#ok

            if l > bestSNR
                bestSNR = l;
                bestEpoch = jj;
                filename = 'params.mat';
                save(filename,'learnables','states');
            end
        end

        iteration = iteration + 1;

        % Get a new batch of training data
        [x1Batch,x2Batch,mixBatch] = next(mqueue);
        x1Batch = reshape(x1Batch,[duration 1 miniBatchSize]);
        x2Batch = reshape(x2Batch,[duration 1 miniBatchSize]);
        mixBatch = reshape(mixBatch,[duration 1 miniBatchSize]);

        x1Batch = dlarray(x1Batch,'SCB');
        x2Batch = dlarray(x2Batch,'SCB');
        mixBatch = dlarray(mixBatch,'SCB');

        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            x1Batch = gpuArray(x1Batch);
            x2Batch = gpuArray(x2Batch);
            mixBatch = gpuArray(mixBatch);
        end

        % Evaluate the model gradients and loss using dlfeval and the modelGradients function.
        [gradients,states] = dlfeval( @modelGradients,mixBatch,x1Batch,x2Batch,learnables,states,miniBatchSize);

        % Update the network parameters using the ADAM optimizer.
        [learnables,averageGrad,averageSqGrad] = adamupdate(learnables,gradients,averageGrad,averageSqGrad,iteration,learnRate,gradDecay,sqGradDecay);
        
    end

    % Reduce the learning rate if the validation accuracy did not improve
    % during the epoch
    if bestEpoch ~= jj
        learnRate = learnRate/2;
    end
end

%% 
% Plot the validation SNR values.

if ~reduceDataset
    valIterNum = 0:length(valSNR)-1;
    figure
    semilogx(numIterPerValidation*(valIterNum-1),valSNR,'b*-')
    grid on
    xlabel('Iteration #')
    ylabel('Validation SINR (dB)')
    valFig.Visible = 'on';
end
%% References
% [1] Yi Luo, Nima Mesgarani, "Conv-tasnet: Surpassing ideal time–frequency 
% magnitude masking for speech separation," 2019 IEEE/ACM transactions on audio, 
% speech, and language processing, vol. 29, issue 8, pp. 1256-1266.
% 
% [2] V. Panayotov, G. Chen, D. Povey and S. Khudanpur, "Librispeech: An ASR 
% corpus based on public domain audio books," 2015 IEEE International Conference 
% on Acoustics, Speech and Signal Processing (ICASSP), Brisbane, QLD, 2015, pp. 
% 5206-5210, doi: 10.1109/ICASSP.2015.7178964
%% Supporting Functions
% 

function [mixDatastore,speaker1Datastore,speaker2Datastore] = createTrainingDataset(netFolder,datasetFolder,downloadDatasetFolder,reduceDataset,miniBatchSize)
% createTrainingDataset Create training dataset

newDatasetPath = fullfile(downloadDatasetFolder,'speech-sep-dataset');

%Create the new dataset folders if they do not exist already.
processDataset = ~isfolder(newDatasetPath);
if processDataset
    mkdir(newDatasetPath);
    mkdir([newDatasetPath '/sp1']);
    mkdir([newDatasetPath '/sp2']);
    mkdir([newDatasetPath '/mix']);
end

%Create an audioDatastore that points to the LibriSpeech dataset.
if reduceDataset
    ads = audioDatastore([repmat({fullfile(netFolder,'speaker1.wav')},1,4),...
                          repmat({fullfile(netFolder,'speaker2.wav')},1,4)]);
else
    ads = audioDatastore(datasetFolder,'IncludeSubfolders',true);
end

% The LibriSpeech dataset is comprised of signals from different speakers.
% The unique speaker ID is encoded in the audio file names.

% Extract the speaker IDs from the file names.
if reduceDataset
    ads.Labels = categorical([repmat({'1'},1,4),repmat({'2'},1,4)]);
else
    ads.Labels = categorical(extractBetween(ads.Files,fullfile(datasetFolder,filesep),filesep));
end

% You will create mixture signals comprised of utterances of two random speakers.  
% Randomize the IDs of all the speakers.
names = unique(ads.Labels);
names = names(randperm(length(names)));

% In this example, you create training signals based on 400 speakers. You
% generate mixture signals based on combining utterances from 200 pairs of
% speakers. 

% Define the two groups of speakers.
numPairs = min(200,floor(numel(names)/2)); 
n1 = names(1:numPairs);
n2 = names(numPairs+1:2*numPairs);

% Create the new dataset. For each pair of speakers: 
% * Use subset to create two audio datastores, each containing files
%   corresponding to their respective speaker.
% * Adjust the datastores so that they have the same number of files.
% * Combine the two datastores using combine. 
% * Use writeall to preprocess the files of the combined datastore and write
%   the new resulting signals to disk.

% The preprocessing steps performed to create the signals before writing
% them to disk are encapsulated in the function createTrainingFiles. For
% each pair of signals:
% * You downsample the signals from 16 kHz to 8 kHz. 
% * You randomly select 4 seconds from each downsampled signal. 
% * You create the mixture by adding the 2 signal chunks.
% * You adjust the signal power to achieve a randomly selected
%   signal-to-noise value in the range [-5,5] dB.
% * You write the 3 signals (corresponding to the first speaker, the second
%   speaker, and the mixture, respectively) to disk.
parfor index=1:length(n1)
    spkInd1 = n1(index);
    spkInd2 = n2(index);
    spk1ds = subset(ads,ads.Labels==spkInd1);
    spk2ds = subset(ads,ads.Labels==spkInd2);
    L = min(length(spk1ds.Files),length(spk2ds.Files));
    L = floor(L/miniBatchSize) * miniBatchSize;
    spk1ds = subset(spk1ds,1:L);
    spk2ds = subset(spk2ds,1:L);
    pairds = combine(spk1ds,spk2ds);
    writeall(pairds,newDatasetPath,'FolderLayout','flatten','WriteFcn',@(data,writeInfo,outputFmt)createTrainingFiles(data,writeInfo,outputFmt,reduceDataset));
end

% Create audio datastores pointing to the files corresponding to the individual speakers and the mixtures.
mixDatastore = audioDatastore(fullfile(newDatasetPath,'mix'));
speaker1Datastore = audioDatastore(fullfile(newDatasetPath,'sp1'));
speaker2Datastore = audioDatastore(fullfile(newDatasetPath,'sp2'));
end

function mix = createTrainingFiles(data,writeInfo,~,varargin)
% createTrainingFiles - Preprocess the training signals and write them to disk

reduceDataset = varargin{1};

duration = 4*8000;

x1 = data{1};
x2 = data{2};

% Resample from 16 kHz to 8 kHz
if ~reduceDataset
    x1 = resample(x1,1,2);
    x2 = resample(x2,1,2);
end

% Read a chunk from the first speaker signal
if length(x1)<=duration
    x1 = [x1;zeros(duration-length(x1),1)];
else
    startInd = randi([1 length(x1)-duration],1);
    endInd = startInd + duration - 1;
    x1 = x1(startInd:endInd);
end

% Read a chunk from the second speaker signal
if length(x2)<=duration
    x2 = [x2;zeros(duration-length(x2),1)];
else
    startInd = randi([1 length(x2)-duration],1);
    endInd = startInd + duration - 1;
    x2 = x2(startInd:endInd);
end

x1 = x1./max(abs(x1));
x2 = x2./max(abs(x2));

% SNR [-5 5] dB
s = snr(x1,x2);
targetSNR = 10 * (rand - 0.5);
x1b = 10^((targetSNR-s)/20) * x1;
mix = x1b + x2;
mix = mix./max(abs(mix));

if reduceDataset
    [~,n] = fileparts(tempname);
    name = sprintf('%s.wav',n);
else
    [~,s1] = fileparts(writeInfo.ReadInfo{1}.FileName);
    [~,s2] = fileparts(writeInfo.ReadInfo{2}.FileName);
    name = sprintf('%s-%s.wav',s1,s2);
end

audiowrite(sprintf('%s',fullfile(writeInfo.Location,'sp1',name)),x1,8000);
audiowrite(sprintf('%s',fullfile(writeInfo.Location,'sp2',name)),x2,8000);
audiowrite(sprintf('%s',fullfile(writeInfo.Location,'mix',name)),mix,8000);

end

function [grad, states] = modelGradients(mix,x1,x2,learnables,states,miniBatchSize)
% modelGradients Compute the model gradients

[y1,y2,states] = separateSpeakers(mix,learnables,states,true);

m = uPIT(x1,y1,x2,y2);
l = sum(m);
loss = -l./miniBatchSize;

grad = dlgradient(loss,learnables);

end

function m = uPIT(x1,y1,x2,y2)
% uPIT - Compute utterance-level permutation invariant training
v1 = SISNR(y1,x1);
v2 = SISNR(y2,x2);
m1 = mean([v1;v2]);

v1 = SISNR(y2,x1);
v2 = SISNR(y1,x2);
m2 = mean([v1;v2]);

m = max(m1,m2);
end

function z = SISNR(x,y)
% SISNR - Compute SI-SNR
x = x - mean(x);
y = y - mean(y);

t = sum(x.*y) .* y ./ (sum(y.^2)+eps);
n = x - t;

z = 20*log((sqrt(sum(t.^2))+eps)./sqrt((sum(n.^2))+eps))/log(10);

end

function [learnables,states] = initializeNetworkParams
% initializeNetworkParams - Initialize the learnables and states of the
% network
learnables.Conv1W = initializeGlorot(20,1,256);
learnables.Conv1B = dlarray(zeros(256,1,'single'));

learnables.ln_weight = dlarray(ones(1,256,'single'));
learnables.ln_bias = dlarray(zeros(1,256,'single'));

learnables.Conv2W = initializeGlorot(1,256,256);
learnables.Conv2B = dlarray(zeros(256,1,'single'));

for index=1:32
    blk = [];
    blk.Conv1W = initializeGlorot(1,256,512);
    blk.Conv1B = dlarray(zeros(512,1,'single'));
    blk.Prelu1 = dlarray(single(0.25));
    blk.BN1Offset = dlarray(zeros(512,1,'single'));
    blk.BN1Scale = dlarray(ones(512,1,'single'));
    blk.Conv2W = initializeGlorot(3,1,512);
    blk.Conv2W =  reshape(blk.Conv2W,[3 1 1 512]);
    blk.Conv2B = dlarray(zeros(512,1,'single'));
    blk.Prelu2 = dlarray(single(0.25));
    blk.BN2Offset= dlarray(zeros(512,1,'single'));
    blk.BN2Scale= dlarray(ones(512,1,'single'));
    blk.Conv3W = initializeGlorot(1,512,256);
    blk.Conv3B = dlarray(ones(256,1,'single'));

    learnables.Blocks(index) = blk;

    s = [];
    s.BN1Mean= dlarray(zeros(512,1,'single'));
    s.BN1Var= dlarray(ones(512,1,'single'));
    s.BN2Mean = dlarray(zeros(512,1,'single'));
    s.BN2Var = dlarray(ones(512,1,'single'));

    states(index) = s; %#ok
end

learnables.Conv3W = initializeGlorot(1,256,512);
learnables.Conv3B = dlarray(zeros(512,1,'single'));

learnables.TransConv1W = initializeGlorot(20,1,256);
learnables.TransConv1B = dlarray(zeros(1,1, 'single'));

end

function weights = initializeGlorot(filterSize,numChannels,numFilters)
% initializeGlorot - Perform Glorot initialization
sz = [filterSize numChannels numFilters];
numOut = prod(filterSize) * numFilters;
numIn = prod(filterSize) * numFilters;

Z = 2*rand(sz,'single') - 1;
bound = sqrt(6 / (numIn + numOut));

weights = bound * Z;
weights = dlarray(weights);

end

function [output1, output2, states] = separateSpeakers(input, learnables, states, training)
% separateSpeakers - Separate two speaker signals from a mixture input
if ~isa(input,'dlarray')
    input = dlarray(input,'SCB');
end

weights = learnables.Conv1W;
bias = learnables.Conv1B;
x = dlconv(input, weights,bias, 'Stride', 10);

x = relu(x);
x0 = x;

x = x-mean(x, 2);
x = x./sqrt(mean(x.^2, 2) + 1e-5);
x = x.*learnables.ln_weight + learnables.ln_bias;

weights = learnables.Conv2W;
bias = learnables.Conv2B;
encoderOut = dlconv(x, weights, bias);

for index = 1:32
    [encoderOut,s] = convBlock(encoderOut, index-1,learnables.Blocks(index),states(index),training);
    states(index) = s;
end

weights = learnables.Conv3W;
bias = learnables.Conv3B;
masks = dlconv(encoderOut, weights, bias);
masks = relu(masks);

mask1 = masks(:,1:256,:);
mask2 = masks(:,257:512,:);

out1 = x0 .* mask1;
out2 = x0 .* mask2;

weights = learnables.TransConv1W;
bias = learnables.TransConv1B;
output2 = dltranspconv(out1, weights, bias, 'Stride', 10);
output1 = dltranspconv(out2, weights, bias, 'Stride', 10);

if ~training
    output1 = gather(extractdata(output1));
    output2 = gather(extractdata(output2));

    output1 = output1./max(abs(output1));
    output2 = output2./max(abs(output2));
end

end

function [output,state] = convBlock(input, count,learnables,state,training)

% Conv:
weights = learnables.Conv1W;
bias = learnables.Conv1B;
conv1Out = dlconv(input, weights, bias);

% PRelu:
conv1Out = relu(conv1Out) - learnables.Prelu1.*relu(-conv1Out);

% BatchNormalization:
offset = learnables.BN1Offset;
scale = learnables.BN1Scale;
datasetMean = state.BN1Mean;
datasetVariance = state.BN1Var;
if training
    [batchOut, dsmean, dsvar] = batchnorm(conv1Out, offset, scale, datasetMean, datasetVariance);
    state.BN1Mean = dsmean;
    state.BN1Var = dsvar;
else
    batchOut = batchnorm(conv1Out, offset, scale, datasetMean, datasetVariance);
end

% Conv:
weights = learnables.Conv2W;
bias = learnables.Conv2B;
padding = [1 1] * 2^(mod(count,8));
dilationFactor = 2^(mod(count,8));
convOut = dlconv(batchOut, weights, bias,'DilationFactor', dilationFactor, 'Padding', padding);

% PRelu:
convOut = relu(convOut) - learnables.Prelu2.*relu(-convOut);

% BatchNormalization:
offset = learnables.BN2Offset;
scale = learnables.BN2Scale;
datasetMean = state.BN2Mean;
datasetVariance = state.BN2Var;
if training
    [batchOut, dsmean, dsvar] = batchnorm(convOut, offset, scale, datasetMean, datasetVariance);
    state.BN2Mean = dsmean;
    state.BN2Var = dsvar;
else
    batchOut = batchnorm(convOut, offset, scale, datasetMean, datasetVariance);
end

% Conv:
weights = learnables.Conv3W;
bias = learnables.Conv3B;
output = dlconv(batchOut, weights, bias);

% Skip connection
output = output + input;

end

function [speaker1,speaker2] = separateSpeakersTimeFrequency(mix,pathToNet)
% separateSpeakersTimeFrequency - STFT-based speaker separation function
WindowLength  = 128;
FFTLength     = 128;
OverlapLength = 128-1;
win           = hann(WindowLength,"periodic");

% Downsample to 4 kHz
mix = resample(mix,1,2);

P0 = stft(mix, 'Window', win, 'OverlapLength', OverlapLength,...
    'FFTLength', FFTLength, 'FrequencyRange', 'onesided');
P = log(abs(P0) + eps);
MP = mean(P(:));
SP = std(P(:));
P = (P-MP)/SP;

seqLen = 20;
PSeq  = zeros(1 + FFTLength/2,seqLen,1,0);
seqOverlap = seqLen;

loc = 1;
while loc < size(P,2)-seqLen
    PSeq(:,:,:,end+1) = P(:,loc:loc+seqLen-1); %#ok
    loc = loc + seqOverlap;
end

PSeq  = reshape(PSeq, [1 1 (1 + FFTLength/2) * seqLen size(PSeq,4)]);

s = load(fullfile(pathToNet,"CocktailPartyNet.mat"));
CocktailPartyNet = s.CocktailPartyNet;
estimatedMasks = predict(CocktailPartyNet,PSeq);

estimatedMasks = estimatedMasks.';
estimatedMasks = reshape(estimatedMasks,1 + FFTLength/2,numel(estimatedMasks)/(1 + FFTLength/2));

mask1   = estimatedMasks; 
mask2 = 1 - mask1;

P0 = P0(:,1:size(mask1,2));

P_speaker1 = P0 .* mask1;

speaker1 = istft(P_speaker1, 'Window', win, 'OverlapLength', OverlapLength,...
    'FFTLength', FFTLength, 'ConjugateSymmetric', true,...
    'FrequencyRange', 'onesided');
speaker1 = speaker1 / max(abs(speaker1));

P_speaker2 = P0 .* mask2;

speaker2 = istft(P_speaker2, 'Window', win, 'OverlapLength', OverlapLength,...
    'FFTLength',FFTLength, 'ConjugateSymmetric',true,...
    'FrequencyRange', 'onesided');
speaker2 = speaker2 / max(speaker2);

speaker1 = resample(double(speaker1),2,1);
speaker2 = resample(double(speaker2),2,1);
end
%% 
% _Copyright 2021 The MathWorks, Inc._