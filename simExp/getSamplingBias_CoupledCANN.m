% Demonstarte the sampling bias in the coupled CANNs to reproduce the
% figure in Fetsch, Nat. Neurosci., 2012.

% We analyze how the indirect cue weight changes with the indirect input intensity.

% Wen-Hao Zhang, Jan-7-2025
% wenhao.zhang@utsouthwestern.edu
% @UT Southwestern Medical Center

setWorkPath;

% Load parameters
parsCoupledCANNs;

% NetPars.AmplRatio = [0.05:0.01:0.09, 0.1:0.1:1];
% NetPars.AmplRatio = unique([0.2:0.05:0.4, 0.5:0.1:0.8]);
NetPars.AmplRatio = 0.3:0.1: 1.2;
NetPars.AmplRatio = [0.3*ones(size(NetPars.AmplRatio)); NetPars.AmplRatio];
NetPars.cueCond = 0:2; % Don't change the order of cue conditions

NetPars.JrcRatio = 0.5;
% NetPars.JrcRatio = 0.7;
NetPars.JrpRatio = 1;
NetPars.tLen = 2e3;
NetPars.tStat = 1.5e3+1;

NetPars.fanoFactor = 0.5;
NetPars.Posi = 2*[-1;1];

NetPars.seedNois = sum(clock)*100;

% Generate grid of parameters
[parGrid, dimPar] = paramGrid(NetPars);
parGrid = arrayfun(@(x) getDependentPars(x), parGrid);

%% Net Simulation
NetStat = struct('BumpPos', [], ...
    'meanBumpPos', [], ...
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

%% Cue weights (actual values and predictions)

% Prediction of cue 1 weights in both networks
varBumpPos = reshape({NetStat.varBumpPos}, size(parGrid));
varBumpPos = cellfun(@(x) diag(x), varBumpPos, 'UniformOutput', false);
varBumpPos = cell2mat(shiftdim(varBumpPos,-1)); % [Nets, Cue Intensity, Cue conditions]
CueWeight1_Pred = varBumpPos(:,:,3) ./ sum(varBumpPos(:,:, 2:3), 3);

% Actual weight of cue 1 in both networks
meanBumpPos = reshape({NetStat.meanBumpPos}, size(NetStat));
meanBumpPos = cell2mat(shiftdim(meanBumpPos,-1)); % [Net, Cue Intensity, Cue conditions]
CueWeight1 = (meanBumpPos(:,:,1) - meanBumpPos(:,:,3)) ...
    ./ (meanBumpPos(:,:,2) - meanBumpPos(:,:,3));


% Prediction of cue 1's variance
CueVar1_Pred = 1./sum(1./varBumpPos(:,:, 2:3), 3);
CueVar1 = varBumpPos(:,:,1);

% -------------------------------------------
% Find the effective prior in the circuit
Prior_precision = zeros(1, numel(parGrid));
for iterPar = 1: numel(NetStat)
    Omega = inv(NetStat(iterPar).varBumpPos); % Posterior precision matrix
    Prior_precision(iterPar) = -Omega(1,2);
end
Prior_precision = reshape(Prior_precision, size(NetStat));
clear Omega iterPar


% -------------------------------------------
% Bump height and additivity index
OHeight = reshape({NetStat.OHeightAvg}, size(parGrid));
OHeight = cellfun(@(x) x(1), OHeight);
AddIdx = OHeight(:,1) ./ sum(OHeight(:,2:3), 2);


%% The change of cue weight with the change of indirect cue input intensity
% 
% The mean and variance of bump position in network 1
figure('position',[500 500 800 500])
% figure
subplot(2,3,1)
plot(NetPars.AmplRatio(2,:), squeeze(meanBumpPos(1,:,:)))
ylabel('Mean of bump position')
% title('Net 1')
subplot(2,3,4)
plot(NetPars.AmplRatio(2,:), squeeze(varBumpPos(1,:,:)))
ylabel('Var of bump position')
xlabel('Cue 2 intensity')

% The effective prior precision in the circuit
subplot(2,3,2)
plot(Prior_precision)
xlabel('Cue 2 intensity')
ylabel('Prior precision')
legend('Combined', 'Cue 1', 'Cue 2')


% Plot the firing rate with cue intensities
subplot(2,3,5)
yyaxis left
plot(NetPars.AmplRatio(2,:), OHeight)
% axis square
ylabel('Rate (net 1)')
xlabel('Cue 2 intensity')

yyaxis right
plot(NetPars.AmplRatio(2,:), AddIdx)
ylabel('Additivity index')

legend('Comb', 'Cue 1', 'Cue 2')


subplot(2,3,3)
plot(NetPars.AmplRatio(2,:), CueWeight1(1,:))
hold on
plot(NetPars.AmplRatio(2,:), CueWeight1_Pred(1,:))
xlabel('Cue 2 intensity')
ylabel('Direct cue weight')
legend('Actual', 'Prediction')



%% Plot

% % The mean and variance of bump position in network 1
% figure('position',[600 500 400 400])
% subplot(2,2,1)
% plot(NetPars.AmplRatio(2,:), squeeze(meanBumpPos(1,:,:)))
% ylabel('Mean of bump position')
% title('Net 1')
% subplot(2,2,3)
% plot(NetPars.AmplRatio(2,:), squeeze(varBumpPos(1,:,:)))
% ylabel('Var of bump position')
% xlabel('Cue 2 intensity')
% 
% subplot(2,2,2)
% plot(NetPars.AmplRatio(2,:), squeeze(meanBumpPos(2,:,:)))
% ylabel('Mean of bump position')
% title('Net 2')
% subplot(2,2,4)
% plot(NetPars.AmplRatio(2,:), squeeze(varBumpPos(2,:,:)))
% ylabel('Var of bump position')
% xlabel('Cue 2 intensity')
% legend('Combined', 'Cue 1', 'Cue 2')
% 
% 
% figure
% for IdxNet = 1:2
% 
%     subplot(2,2,IdxNet)
%     yyaxis left
%     plot(NetPars.AmplRatio(2,:), 1- CueWeight1(IdxNet,:))
%     hold on
%     plot(NetPars.AmplRatio(2,:), 1- CueWeight1_Pred(IdxNet,:))
%     xlabel('Visual reliability')
%     ylabel('Vestibular weight')
%     axis square
% 
%     yyaxis right
%     plot(NetPars.AmplRatio(2,:),  CueWeight1_Pred(1,:)- CueWeight1(1,:))
%     ylabel('Diff. vest. weight (Obs. - Pred.)')
% 
%     legend('Observed', 'Prediction', 'Diff')
% 
%     title(['Net ' num2str(IdxNet)]);
% 
%     % Plot the firing rate with cue intensities
%     OHeight = reshape({NetStat.OHeightAvg}, size(parGrid));
%     OHeight = cellfun(@(x) x(IdxNet), OHeight);
% 
%     subplot(2,2,2+IdxNet)
%     yyaxis left
%     plot(NetPars.AmplRatio(2,:), OHeight)
%     axis square
%     legend('Comb', 'Cue 1', 'Cue 2')
%     ylabel(['Rate (net'  num2str(IdxNet) ')'])
%     xlabel('Cue 1 intensity')
% 
%     yyaxis right
%     AddIdx = OHeight(:,1) ./ sum(OHeight(:,2:3), 2);
%     plot(NetPars.AmplRatio(2,:), AddIdx)
%     ylabel('Additivity index')
% 
%     %     clear OHeight
% end


%% Bias in the network 1

% Plot the firing rate with cue intensities
% figure
% 
% for IdxNet = 1:2
% 
%     OHeight = reshape({NetStat.OHeightAvg}, size(parGrid));
%     OHeight = cellfun(@(x) x(IdxNet), OHeight);
%     AddIdx = OHeight(:,1) ./ sum(OHeight(:,2:3), 2);
% 
%     subplot(2,3,3*IdxNet-2)
%     yyaxis left
%     plot(NetPars.AmplRatio(2,:),  CueWeight1_Pred(IdxNet,:)- CueWeight1(IdxNet,:))
%     xlabel('Visual reliability')
%     ylabel('Bias of vest. weight')
%     yyaxis right
%     plot(NetPars.AmplRatio(2,:), AddIdx)
%     ylabel('Additivity index')
%     axis square
% 
% 
%     subplot(2,3,3*IdxNet-1)
%     VarRatio = CueVar1./ CueVar1_Pred - 1;
% 
%     yyaxis left
%     plot(NetPars.AmplRatio(2,:), VarRatio(1,:))
%     xlabel('Visual reliability')
%     ylabel('Bias of var. (%)')
% 
%     yyaxis right
%     plot(NetPars.AmplRatio(2,:), AddIdx)
%     ylabel('Additivity index')
%     axis square
%     title(['Net ' num2str(IdxNet)]);
% 
%     subplot(2,3,3*IdxNet)
%     yyaxis left
%     plot(AddIdx, CueWeight1_Pred(IdxNet,:)- CueWeight1(IdxNet,:))
%     xlabel('Additivity index')
%     ylabel('Bias of vest. weight')
%     yyaxis right
%     plot(AddIdx, VarRatio(IdxNet,:))
%     ylabel('Bias of var. (%)')
%     axis square
% end
