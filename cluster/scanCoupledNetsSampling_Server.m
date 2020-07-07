% Test the sampling-based inference in the decentralized system under different parameters

% Wen-Hao Zhang, May 21, 2020
% wenhao.zhang@pitt.edu

setWorkPath;

% Load parameters
parsCoupledCANNs;

NetPars.fanoFactor = 0.5;

NetPars.tTrial  = 500 * NetPars.tau;
NetPars.nTrials = 10; % number of trials
NetPars.tLen    = NetPars.nTrials * NetPars.tTrial;

nMCSim = 50; % Number of Monte-Carlo simulations per network size
numNetsArray = [2:5, 10]; % Number of networks in the distributed system

NetPars = arrayfun(@(x) getDependentPars(x), NetPars);

%% Net Simulation
NetStat = struct('meanBumpPos', [], ...
    'varBumpPos', [], ...
    'OHeightAvg', [], ...
    'OAvgXTime', [], ...
    'OStdXTime', [], ...
    'UHeightAvg', []);
NetStat = repmat(NetStat, [nMCSim, length(numNetsArray)]);

JMatArray = cell(nMCSim, length(numNetsArray));
InputPosi = cell(nMCSim, length(numNetsArray));
InputAmpl = cell(nMCSim, length(numNetsArray));

tStart = clock;

parpool(4);
parfor iterMCSim = 1: nMCSim * length(numNetsArray)
    fprintf('Progress: %d/%d\n', iterMCSim, nMCSim*length(numNetsArray));
    netpars = NetPars;
    
    netpars.numNets = numNetsArray(ceil(iterMCSim/nMCSim));
    
    % Randomly generate a connection matrix
    rng(sum(clock)*100); % I need to randomly set the random seed. The simNet code will set the net random seed.
    JMat = rand(netpars.numNets)* netpars.Jc;
    % JMat = JMat/ netpars.numNets; % Scale with the number of networks
    % JMat = (JMat + JMat')/2;
    JMatArray{iterMCSim} = JMat;
    netpars.JMat = JMat;
    
    % Randomly generate input location
    netpars.Posi = rand(netpars.numNets,1)*10 - 5;
    InputPosi{iterMCSim} = netpars.Posi;
    
    % Randomly generate input intensity
    netpars.AmplRatio = 1.5*rand(netpars.numNets,1);
    InputAmpl{iterMCSim} = netpars.AmplRatio;
    
    netpars = arrayfun(@(x) getDependentPars(x), netpars);
    % ---------------------------------------------------------------
    % Network input
    InputSet = makeNetInput([], netpars);
    
    % Run simulation
    % outArgs = struct('InputSet', [], 'NetStat', NetStat(iterMCSim));
    outArgs = struct('NetStat', NetStat(iterMCSim));
    [InputSet, NetStat(iterMCSim)] = simCoupledAttractors1(InputSet, netpars, outArgs);
end

tEnd = clock;

%% Save
savePath = fullfile(Path_RootDir, 'Data');
mkdir(savePath);

str = datestr(now, 'yymmddHHMM');
fileName = ['scanNetSampling_', str(1:6), '_', str(7:end) '.mat'];

save(fullfile(savePath, fileName), '-v7.3')

