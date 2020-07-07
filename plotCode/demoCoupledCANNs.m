% Test the performance of decentralized system under different parameters

% Wen-Hao Zhang, Oct-6-2016
% wenhaoz1@andrew.cmu.edu
% @Carnegie Mellon University

setWorkPath;

% Load parameters
parsCoupledCANNs;

NetPars.JrpRatio = 0:0.1:1; 
% NetPars.AmplRatio = 0.2:0.2:2;
NetPars.AmplRatio = 0.5;
NetPars.AmplRatio = repmat(NetPars.AmplRatio, [NetPars.numNets, 1]);

% Generate grid of parameters
[parGrid, dimPar] = paramGrid(NetPars);
parGrid = arrayfun(@(x) getDependentPars(x), parGrid);


%% Net Simulation
NetStat = struct('meanBumpPos', [], ...
    'mrlBumpPos', [], ...
    'concBumpPos', [], ...
    'varBumpPos', [], ...
    'OHeightAvg', []);
NetStat = repmat(NetStat, size(parGrid));

tStart = clock;
for iterPar = 1: numel(parGrid)
    fprintf('Progress: %d/%d\n', iterPar, numel(parGrid));
    netpars = parGrid(iterPar);
    
    % Network input
    InputSet = makeNetInput([], netpars);
    
    % Run simulation
    outArgs = struct('InputSet', [], 'NetStat', NetStat(iterPar));
    [InputSet, NetStat(iterPar)] = simCoupledAttractors(InputSet, netpars, outArgs);
end

tEnd = clock;

%% Theoretical prediction

InfoFisher = sqrt(2*pi) * [parGrid.Ampl] .* NetPars.rho./ (sqrt(2)*NetPars.TunWidth);


%%
figure
yyaxis left
plot(NetPars.AmplRatio', 1./[NetStat.varBumpPos]')
hold on
plot(NetPars.AmplRatio', InfoFisher')

xlabel('Input intensity')
title(['JrcRatio=', num2str(NetPars.JrcRatio)])

yyaxis right
plot(NetPars.AmplRatio', [NetStat.OHeightAvg]')
% varNetSim

legend('Bump posi. var.', 'Fisher info.', 'Rate average','location', 'best')