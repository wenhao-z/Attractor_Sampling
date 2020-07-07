% Test the performance of decentralized system under different parameters

% Wen-Hao Zhang, Oct-6-2016
% wenhaoz1@andrew.cmu.edu
% @Carnegie Mellon University

setWorkPath;

% Load parameters
parsCoupledCANNs;

NetPars.numNets = 5;
NetPars.AmplRatio = 1*ones(NetPars.numNets,1);

NetPars.fanoFactor = 0.5;
% NetPars.Posi = 2*[-1;0;1];

nMCSim = 50;

NetPars = arrayfun(@(x) getDependentPars(x), NetPars);

%% Net Simulation
NetStat = struct('meanBumpPos', [], ...
    'varBumpPos', [], ...
    'OHeightAvg', [], ...
    'OAvgXTime', [], ...
    'OStdXTime', [], ...
    'UHeightAvg', []);
NetStat = repmat(NetStat, [1, nMCSim]);

JMatArray = zeros(NetPars.numNets, NetPars.numNets, nMCSim);
InputPosi = zeros(NetPars.numNets, nMCSim);

tStart = clock;
% for iterPar = 1: numel(parGrid)
for iterPar = 1: nMCSim
    fprintf('Progress: %d/%d\n', iterPar, nMCSim);
    
    % Randomly generate a connection matrix
    rng(sum(clock)*100); % I need to randomly set the random seed. The simNet code will set the net random seed.
    JMat = rand(NetPars.numNets)* NetPars.Jc;
    % JMat = (JMat + JMat')/2;
    JMatArray(:,:,iterPar) = JMat;
    NetPars.JMat = JMat;
    
    % Randomly generate input location 
    % randPosi = 5*rand(1); % random distribution in [0,5];
    % NetPars.Posi = [randPosi; 0; -randPosi];
    NetPars.Posi = rand(NetPars.numNets,1)*10-5;
    InputPosi(:,iterPar) = NetPars.Posi;
    
    % ---------------------------------------------------------------
    % Network input
    InputSet = makeNetInput([], NetPars);
    
    % Run simulation
    outArgs = struct('InputSet', [], 'NetStat', NetStat(iterPar));
    [InputSet, NetStat(iterPar)] = simCoupledAttractors1(InputSet, NetPars, outArgs);
end

tEnd = clock;

%% Parameters of sample distributions
PreMat_Sample = {NetStat.varBumpPos};
PreMat_Sample = cellfun(@inv, PreMat_Sample, 'uniformout', 0);
PreMat_Sample = reshape(cell2mat(PreMat_Sample), NetPars.numNets, NetPars.numNets, []);

%% Theoretical prediction

wfwd = 8/3^(3/2) * NetPars.fanoFactor * 2.5;

% New version includes the spatial convolution of the feedforward input.
PreMat_LH = 2*sqrt(pi) * diag(NetPars.Ampl) / NetPars.TunWidth/ wfwd;

% The variance of noise is the synaptic input bump
varBumpPosTheory = 4*NetPars.fanoFactor*NetPars.TunWidth/sqrt(pi)/ 3^(3/2) ./ [NetPars.Ampl];

% ------------------------------------------------------
% Prediction of mean of samples
% Find the prior precision which best explains the sample distribution
meanSampleTheory = zeros(NetPars.numNets, nMCSim);
Lambda_s = zeros(1, nMCSim);
KLD = zeros(1, nMCSim);

for iterPar = 1: numel(NetStat)
    meanSampleTheory(:,iterPar) = NetStat(iterPar).varBumpPos * ...
        PreMat_LH * InputPosi(:,iterPar); %NetPars.Posi;
%     [Lambda_s(iterPar), KLD(iterPar)] = findPriorPrecisionHD(NetStat(iter).meanBumpPos, NetStat(iter).varBumpPos, ...
%         NetPars.Posi, PreMat_LH);
end
meanSampleTheory = reshape(meanSampleTheory, [NetPars.numNets, nMCSim]);

%%
figure

subplot(1,2,1)
hold on
plot(reshape([NetStat.meanBumpPos],[],1), meanSampleTheory(:), 'o', 'markersize', 4)
xyLim = max(abs(InputPosi(:)));
plot(xyLim*[-1,1], xyLim*[-1,1], '--k')
xlabel('Mean of samples')
ylabel('Mean (theory)')
axis square

subplot(1,2,2)
hold on
for iterPar = 1: nMCSim
    Omega = inv(NetStat(iterPar).varBumpPos); % Posterior precision matrix
    
    xDat = -reshape(triu(Omega,1), 1,[]);
    %     yDat = reshape(triu(JMatArray(:,:,iterPar),1).* NetStat(iterPar).OHeightAvg', 1,[])...
    %         .*NetPars.rho*sqrt(2*pi)/NetPars.TunWidth/wfwd;
    
    % Considering the 
    yDat = JMatArray(:,:,iterPar).* NetStat(iterPar).OHeightAvg';
    yDat = (yDat + yDat')/2;
    yDat = reshape(triu(yDat,1), 1,[]).*NetPars.rho*sqrt(2*pi)/NetPars.TunWidth/wfwd;
    
    plot(xDat(xDat~=0), yDat(xDat~=0), 'o', 'markersize', 4)
%     plot(-reshape(triu(Omega,1), 1,[]), ...
%         reshape(triu(JMatArray(:,:,iterPar),1).* NetStat(iterPar).OHeightAvg', 1,[])...
%         .*NetPars.rho*sqrt(2*pi)/NetPars.TunWidth/wfwd, 'o')
end
clear xDat yDat

xyLim = axis(gca);
xyLim = [min(xyLim([1,3])), max(xyLim([2,4]))];
plot(xyLim, xyLim, '--k')
axis([xyLim, xyLim])

xlabel('Prior precision (searched)')
ylabel('Prior precision (theory)')
axis square
clear xyLim