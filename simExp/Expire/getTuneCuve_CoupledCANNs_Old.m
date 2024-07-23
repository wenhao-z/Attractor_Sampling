% Test the performance of decentralized system under different parameters

% Wen-Hao Zhang, June 2, 2022
% wenhao.zhang@utsouthwestern.edu
% UT Southwestern Medical Center

setWorkPath;

% Load parameters
parsCoupledCANNs;

% Coupling weight
NetPars.JrcRatio = 0.3;
% NetPars.JrpRatio = [0.001, 0.5, 1];
NetPars.JrpRatio = [0.1, 0.2, 0.3, 0.5, 0.8, 1];
% NetPars.JrpRatio = [0.3, 0.5];
NetPars.tLen = 150 * NetPars.tau;


% Bivariate stimuli values to measure bimodal tuning curves
Posi = -180: 45: 180;
PosiArray = zeros([length(Posi)*ones(1,2),2]);
[PosiArray(:,:,1), PosiArray(:,:,2)] = meshgrid(Posi, Posi);
PosiArray = permute(PosiArray, [3,1,2]);
NetPars.Posi = reshape(PosiArray, 2, []);
clear PosiArray

% ----------- Multi-cue condition ----------- 
% Input intensity
NetPars.AmplRatio = [0.1, 0.2, 0.5, 0.8];
NetPars.AmplRatio = [NetPars.AmplRatio; 0.8* ones(size(NetPars.AmplRatio))];

% Generate grid of parameters
[parGrid, dimPar] = paramGrid(NetPars);
parGrid = arrayfun(@(x) getDependentPars(x), parGrid);

% ----------- Single-cue condition ----------- 

% Only cue 1
NetPars_SingleCue = NetPars;
NetPars_SingleCue.AmplRatio(2,:) = 0;
NetPars_SingleCue.Posi = [Posi; zeros(size(Posi))];

[parGrid1, ~] = paramGrid(NetPars_SingleCue);
parGrid1 = arrayfun(@(x) getDependentPars(x), parGrid1);

% Only cue 2
NetPars_SingleCue = NetPars;
NetPars_SingleCue.AmplRatio(1,:) = 0;
NetPars_SingleCue.Posi = [zeros(size(Posi)); Posi];

[parGrid2, ~] = paramGrid(NetPars_SingleCue);
parGrid2 = arrayfun(@(x) getDependentPars(x), parGrid2);

% ---------------------------------------------
parGrid = cat(3, parGrid, parGrid1, parGrid2);

clear NetPars_SingleCue parGrid1 parGrid2

%% Net Simulation
NetStat = struct('meanBumpPos', [], ...
    'varBumpPos', [], ...
    'OHeightAvg', [], ...
    'OAvgXTime', [], ...
    'OStdXTime', [], ...
    'UHeightAvg', []);
NetStat = repmat(NetStat, size(parGrid));

tStart = clock;
parfor iterPar = 1: numel(parGrid)
    fprintf('Progress: %d/%d\n', iterPar, numel(parGrid));
    netpars = parGrid(iterPar);

    % Network input
    InputSet = makeNetInput([], netpars);

    % Run simulation
    outArgs = struct('InputSet', [], 'NetStat', NetStat(iterPar));
    [InputSet, NetStat(iterPar)] = simCoupledAttractors1(InputSet, netpars, outArgs);
end

tEnd = clock;

%% Parameters of sample distributions
PreMat_Sample = {NetStat.varBumpPos};
PreMat_Sample = cellfun(@inv, PreMat_Sample, 'uniformout', 0);
PreMat_Sample = reshape(cell2mat(PreMat_Sample), NetPars.numNets, NetPars.numNets, []);

%% Theoretical prediction

EmpCorrectFactor = 2.8; % Empirical correction factor to predict the position fluctuation

wfwd = 8/3^(3/2) * NetPars.fanoFactor * EmpCorrectFactor;

% New version includes the spatial convolution of the feedforward input.
PreMat_LH = zeros(NetPars.numNets, NetPars.numNets, numel(parGrid));
for iter = 1: numel(parGrid)
    PreMat_LH(:,:,iter) = 2*sqrt(pi) * diag(parGrid(iter).Ampl) / NetPars.TunWidth/ wfwd;
end

% ------------------------------------------------------
% The variance of noise is the firing rate bump
% varBumpPosTheory = [NetStat.OHeightAvg]./ [NetStat.UHeightAvg]*NetPars.TunWidth ...
%     ./ (2*sqrt(pi) .*[parGrid.Ampl]);

% The variance of noise is the synaptic input bump
varBumpPosTheory = 4*NetPars.fanoFactor*NetPars.TunWidth/sqrt(pi)/ 3^(3/2) ./ [parGrid.Ampl];

% ------------------------------------------------------
% Prediction of mean of samples
% Find the prior precision which best explains the sample distribution
meanSampleTheory = zeros(NetPars.numNets, numel(parGrid));
Lambda_s = zeros(size(parGrid));
KLD = zeros(size(parGrid));

for iter = 1: numel(NetStat)
    meanSampleTheory(:,iter) = NetStat(iter).varBumpPos * ...
        PreMat_LH(:,:,iter) * parGrid(iter).Posi;
    [Lambda_s(iter), KLD(iter)] = findPriorPrecision(NetStat(iter).meanBumpPos, ...
        NetStat(iter).varBumpPos, parGrid(iter).Posi, PreMat_LH(:,:,iter));

%     NetPars.Posi;
end
meanSampleTheory = reshape(meanSampleTheory, [NetPars.numNets, size(parGrid)]);

% Find the prior precision in another way (only work for TWO coupled nets)
% Lambda_s = zeros(size(parGrid));
% for iterPar = 1: numel(NetStat)
%     Omega = inv(NetStat(iterPar).varBumpPos); % Posterior precision matrix
%     Lambda_s(iterPar) = -Omega(1,2);
% end
% clear Omega iterPar

%% Extract the firing rate of an example neuron that prefers 0 deg in network 1
szGrid = size(parGrid);
dimGrid = ndims(parGrid);

rate = {NetStat.OAvgXTime};
rate = cellfun(@(x) x(end/2,1), rate);
rate = reshape(rate, szGrid);
rate = permute(rate, [dimGrid, 1: dimGrid-1]);

lenPosi = length(Posi);
rate_TwoCues = rate(1:lenPosi^2,:,:);
rate_Cue1 = rate(lenPosi^2 + (1: lenPosi), :,:);
rate_Cue2 = rate(lenPosi^2 + lenPosi + (1: lenPosi), :,:);

clear rate

%% Find neural weight
% find the neural weight by least square error
Wneu1 = zeros(szGrid(1:2));
Wneu2 = zeros(szGrid(1:2));

X1 = exp(-(-180:45:180).^2/ (2*NetPars.TunWidth^2));
X1 = repmat(X1(:)', lenPosi, 1);
X2 = X1';

for iter = 1: prod(szGrid(1:2))
%     X1 = repmat(rate_Cue1(:,iter)', lenPosi, 1);
%     X2 = repmat(rate_Cue2(:,iter), 1, lenPosi);

    X = [X1(:), X2(:), ones(numel(X1),1)];

    Y = rate_TwoCues(:,iter);

    A = (X'*X)\(X'*Y);
    Wneu1(iter) = A(1);
    Wneu2(iter) = A(2);
end
clear X1 X2 X Y

%%
figure;

maxRate = max(rate_TwoCues(:));

for iter = 1: prod(szGrid(1:2))
    subplot(szGrid(2),szGrid(1),iter);
    
    ratePlot = reshape(rate_TwoCues(:,iter), 9,9);
    contourf(Posi, Posi, ratePlot)
    set(gca, 'xtick', -180:180:180, 'ytick', -180:180:180)
    axis square
    caxis([0, maxRate])
end

%%
figure
subplot(2,2,1);
% yyaxis left
% plot(NetPars.AmplRatio(1,:), Wneu1)
% yyaxis right
% plot(NetPars.AmplRatio(1,:), mean(Lambda_s(:,:,1:lenPosi^2), 3))
plot(NetPars.JrpRatio, Wneu1)
axis square
xlabel('Jrp')
ylabel('Neural weight 1')

subplot(2,2,2);
plot(NetPars.JrpRatio, Wneu2)
ylabel('Neural weight 2')
axis square


subplot(2,2,3)
plot(NetPars.JrpRatio, mean(Lambda_s(:,:,1:lenPosi^2), 3))
ylabel('Prior correlation')
axis square

subplot(2,2,4)
hold on
avgLambda_s = mean(Lambda_s(:,:,1:lenPosi^2), 3);

for iter = 1: size(NetPars.AmplRatio,2)
    plot(avgLambda_s(iter,:), Wneu2(iter,:), '-o')
end
xlabel('Prior correlation')
ylabel('Neural weight')
axis square