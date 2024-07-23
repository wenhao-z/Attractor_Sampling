function [InputSet, NetStat] = simChaoticAttractors(InputSet, NetPars, outArgs)
% A decentralized system for information integration
% The whole system is composed of several networks, with each is modeled as
% a continuous attractor neural network (CANN)

% (ref. W.H. Zhang. et al., JNS 2016 and W.H. Zhang et al., NIPS 2016)
% Features:
% 1. Intra-network connections are translation-invariant bell shape function.
% 2. Inter-network connections have the same profile but different strength
%     with intra-network connections.

% Author: Wen-Hao Zhang, Mar-13-2017
% wenhaoz1@andrew.cmu.edu
% @Carnegie Mellon University

% Unfold parameters from struct NetPars and InputSet
PrefStim    = NetPars.PrefStim;
Width       = NetPars.Width;
dt          = NetPars.dt;
tau         = NetPars.tau;

if nargin == 2
    outArgs = struct('InputSet', [], 'NetStat', []);
end

%% Connection kernel with unit connection strength
switch NetPars.connFunc
    case 'Gaussian'
        TunWidth  = NetPars.TunWidth;
        KerFt = angle(exp(1i*(PrefStim - PrefStim(1)) *pi/Width))* Width/pi;
        KerFt = exp(-KerFt.^2/(2*TunWidth^2))/(sqrt(2*pi)*TunWidth);
    case 'vonMises'
        TunKappa  = NetPars.TunKappa;
        KerFt = angle(exp(1i*(PrefStim - PrefStim(1)) *pi/Width)); % unit: rad
        KerFt = exp(TunKappa * cos(KerFt) )/(2*pi*besseli(0, TunKappa));
end
KerFt = fft(KerFt);

% Weight matrix
if ~isfield(NetPars, 'JMat')
    % The matrix will be right multiply with neural firing rate R (N-by-2 array);
    JMat = (1 - NetPars.JrpRatio) * diag(ones(1, NetPars.numNets)) ...
        + NetPars.JrpRatio * ones(NetPars.numNets);
    JMat = JMat * NetPars.Jrc;
else
    JMat = NetPars.JMat;
end
JMat = JMat'; % Note: the transpose is because Jmat is right multiply with Isyn on line 87

% Reset the random seed
rng(NetPars.seedNois);

% Random connection part
JMatRand = randn(NetPars.N * NetPars.numNets) / sqrt(NetPars.N*NetPars.numNets);
JMatRand = NetPars.JRand * JMatRand;

%% Initiation
% [N, numNets, Time, nTrials]
sizeU = [NetPars.N, NetPars.numNets, ...
    NetPars.tLen/NetPars.dt, NetPars.nTrials];
if isfield(outArgs, 'InputSet')
    UArray = zeros(sizeU);
    OArray = zeros(sizeU);
end
if isfield(outArgs, 'NetStat')
    BumpPos = zeros(sizeU(2:end));
    OHeight = zeros(size(BumpPos));
    UHeight = zeros(size(BumpPos));
end

%% Iteration
for iterTrial = 1: NetPars.nTrials
    U = zeros(sizeU(1:3));
    O = zeros(sizeU(1:3));
    
    % ------------------------------------------
    % Generate new noise sequence of every trial
    %     InputSet = makeNetInput(InputSet, NetPars, ...
    %         struct('ExtNois', [], 'IntNois', []));
    
    % Add the mean value of background inputs
    Iext = InputSet.Iext + NetPars.AmplBkg;
    
    % -----------------------------------------------------------
    % Iteration over time
    for t = 1: InputSet.szIext(3) - 1
        OFt = fft(O(:,:, t));
        
        % Inputs received by congruent neurons
        ISyn = bsxfun(@times, KerFt,  OFt); % Nx2
        ISyn = ifft(ISyn) * JMat;
        
        ISyn = ISyn + JMatRand * reshape(O(:,:,t),[],1);
        
        % Update
        dU = (-U(:,:,t) + ISyn + Iext) * dt/tau;
        % dU = dU + sqrt(NetPars.fanoFactor * U(:,:,t).*(U(:,:,t)>0)*dt/tau)...
        %    .*randn(NetPars.N, NetPars.numNets);
        
        U(:,:,t+1) = U(:,:,t) + dU;
        
        % Synaptic input --> Firing rate
        Urec = U(:,:,t+1);
        Urec(Urec<0) = 0;
        Urec = Urec.^2;
        divU = NetPars.k * sum(Urec, 1);
        
        O(:,:,t+1) = bsxfun(@rdivide, Urec, 1+divU);
    end
    
    if isfield(outArgs, 'InputSet')
        UArray(:,:,:,iterTrial) = U;
        OArray(:,:,:,iterTrial) = O;
    end
    
    % Make statistics of network's activities
    % Calculate the bump position and height
    if exist('BumpPos', 'var')
        BumpPos(:,:,iterTrial) = statBumpPos(O, NetPars);
    end
    if exist('OHeight', 'var')
        OHeight(:,:,iterTrial) = sum(O,1)/ (sqrt(2*pi)*NetPars.TunWidth * NetPars.rho);
        UHeight(:,:,iterTrial) = sum(U,1)/ (2*sqrt(pi)*NetPars.TunWidth * NetPars.rho);
    end
end

%% Estimate the statistics of network activities
if isfield(outArgs, 'NetStat')
    NetStat = statNetResponse(BumpPos, OHeight, UHeight, O, NetPars, outArgs.NetStat);
end

%% Fold variables into output struct
if isfield(outArgs, 'InputSet')
    InputSet.U = UArray;
    InputSet.O = OArray;
end

