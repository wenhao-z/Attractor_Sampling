function InputSet = makeNetInput(InputSet, NetPars, outArgs)
% Make the inputs applied into decentralized network.

% INPUT variables
% InputSet: a void variable (struct) for initialization.
% NetPars:  a struct stores all parameters of network
% outArgs:  indicate which variables to be calculated and outputed.
%           default is calculating Iext, ExtNois, InitNois and initialize
%           RandomStream

% OUTPUT variable
% The InputSet is a struct contains following fields:
% Iext:      [N, numNets, Time]
% ExtNois:   external noise, which is the same size as Iext, 
%            which has been divided by sqrt(dt)
% IntNois:   internal noise, which is the same size as Iext, 
%            which has been divided by sqrt(dt)
% rsExtNois: RandomStream for external noise
% rsIntNois: RandomStream for internal noise

% Wen-hao Zhang, Dec-30-2016
% wenhaoz1@andrew.cmu.edu
% @Carnegie Mellon University

if nargin == 2
    outArgs = struct('Iext', [], 'ExtNois', [], 'IntNois', [], ...
        'initRandStream', []);
end

% The size of external input
szIext = [NetPars.N, NetPars.numNets, NetPars.tLen/NetPars.dt];

%% Set RandStream for external and internal noise
% if isfield(outArgs, 'initRandStream')
%     if isfield(NetPars, 'seedNois')
%         seedNois = NetPars.seedNois;
%     else
%         seedNois = sum(clock*100);
%     end
%     [rsExtNois, rsIntNois] = RandStream.create('mrg32k3a',...
%         'NumStreams',2, 'seed', seedNois);
%     
%     % Fold variables into output struct
%     InputSet.rsExtNois = rsExtNois;
%     InputSet.rsIntNois = rsIntNois;
% end

%% Inputs without noise
if isfield(outArgs, 'Iext')
    N       = NetPars.N;
    Width   = NetPars.Width;
    Ampl    = NetPars.Ampl;
    POSI    = NetPars.Posi;
      
    POSI = shiftdim(POSI, -1); % 1xnumNets)xT
    POSI = repmat(POSI, [N, 1, 1]); % NxnumNetsxT
    POSI = bsxfun(@minus, POSI, NetPars.PrefStim);
    
    switch NetPars.connFunc
        case 'Gaussian'
            % Add periodic condition
            POSI = angle(exp(1i*POSI * pi/Width)) * Width/pi;
            Iext = exp(-(POSI).^2/ (4*NetPars.TunWidth^2)); % [N, numNets, Time]
            Iext = bsxfun(@times, Ampl', Iext);
        case 'vonMises'
            POSI = angle(exp(1i*POSI * pi/Width)); % radian
            Iext = exp(NetPars.TunKappa/2 *(cos(POSI)-1) ); % [N, numNets, Time]
            Iext = bsxfun(@times, Ampl', Iext);
    end
    
    % Apply cueing condition
    switch NetPars.cueCond
        case 1 % Cue 1
            Iext(:, 2, :) = 0;
            % Iext(:, 2:2:end, :) = zeros(size(Iext)./[1,2,1]);
        case 2 % Cue 2
            Iext(:, 1, :) = 0;
    end
    
    % Fold variables into output struct
    InputSet.Iext = Iext;
    InputSet.szIext = szIext; % The desired size of Iext (used in static Iext)
end

%% External noise
% if isfield(outArgs, 'ExtNois')
%     ExtNois = randn(szIext) / sqrt(NetPars.dt);
%     
%     switch NetPars.typeExtNois
%         case 'Gaussian'
%             ExtNois = bsxfun(@times, shiftdim(NetPars.stdExtNois, -1), ExtNois);
%         case 'Poisson'
%             ExtNois = bsxfun(@times, sqrt(NetPars.fanoFactor*InputSet.Iext), ExtNois);
%             % ExtNois = sqrt(NetPars.fanoFactor*InputSet.Iext) .* ExtNois;
%     end
%     
%     % Apply cue condition
%     switch NetPars.cueCond
%         case 1 % Cue 1
%             ExtNois(:, 2:2:end, :) = 0;
%         case 2 % Cue 2
%             ExtNois(:, 1:2:end, :) = 0;
%     end
% 
%     % Fold variables into output struct
%     InputSet.ExtNois = ExtNois;
% end

%% Internal noise
% if isfield(outArgs, 'IntNois')
%     IntNois = randn(InputSet.rsIntNois, szIext) / sqrt(NetPars.dt);
%     IntNois = bsxfun(@times, shiftdim(NetPars.stdIntNois, -1), IntNois);
%     
%     % Fold variables into output struct
%     InputSet.IntNois = IntNois;
% end

