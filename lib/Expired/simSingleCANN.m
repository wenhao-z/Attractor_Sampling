function [InputSet, NetStat] = simSingleCANN(InputSet, NetPars, outArgs)
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

%% Initiation
% [N, numNets, Time, nTrials]
sizeU = [NetPars.N, NetPars.numNets, NetPars.tLen/NetPars.dt, NetPars.nTrials];
if isfield(outArgs, 'InputSet')
    UArray = zeros(sizeU);
    OArray = zeros(sizeU);
end
if isfield(outArgs, 'NetStat')
    BumpPos = zeros(sizeU(2:end));
    OHeight = zeros(size(BumpPos));
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
        OFt = fft(O(:,1,t));
        
        % Inputs received by congruent neurons
        ISyn = bsxfun(@times, KerFt,  OFt); % Nx2
        ISyn = ifft(ISyn) * NetPars.Jrc;
        
        % Update
        dU = (-U(:,:,t) + ISyn + Iext) * dt/tau;
        dU = dU + sqrt(NetPars.fanoFactor * O(:,:,t)*dt/tau).*randn(NetPars.N,1);
        
        U(:,:,t+1) = U(:,:,t) + dU;
        
        % Synaptic input --> Firing rate 
        Urec = U(:,:,t+1);
        Urec(Urec<0) = 0;
        Urec = Urec.^2;
        divU = NetPars.k * sum(Urec, 1);
               
        O(:,:,t+1) = Urec ./ (1 + divU);
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
        OHeight(:,:,iterTrial) = sum(O, 1)/ (sqrt(2*pi)*NetPars.TunWidth * NetPars.rho);    
    end
end

%% Estimate the statistics of network activities
if isfield(outArgs, 'NetStat')
    NetStat = statNetResponse(BumpPos, OHeight, O, NetPars, outArgs.NetStat);
end

%% Fold variables into output struct
if isfield(outArgs, 'InputSet')
    InputSet.U = UArray;
    InputSet.O = OArray;
end

