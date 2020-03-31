clear all
%% preprocess dataset for our method (spectral)

addpath('./Utils/');

% read adequate files
sourcepath = '../../../../media/donati/Data1/Datasets/';  %data path
dataset = 'SHREC_r' %'SCAPE_r'  %'FAUST_r'  %'SHREC';  %name of the dataset
filetype = 'off';  %supported : off, obj, ply and pts + {cmn triangles}
tri_name = 'TRIV_surreal.txt';  %only for pts files

datafolder = [sourcepath dataset '/' filetype '/']
files = dir(fullfile([datafolder ''], ['*.' filetype]));
n_files = length(files)

if filetype == 'pts'  % read common triangulation beforehand
    Tname = [sourcepath dataset tri_name];
    T = load(Tname);
end

% start pre-processing
for pts = 1:n_files
    %pts = 20;
    
    Vname = [datafolder files(pts).name];
    sourcename = split(files(pts).name, '.'); % split to get rif of file extension
    sourcename = sourcename{1}  % print the name
    
    %fid = fopen(Vname);
    if filetype == 'off'
        [V,T] = readOff(Vname);
    end
    if filetype == 'obj'
        [V,T] = readObj(Vname);
    end
    if filetype == 'ply'
        [V,T] = read_ply(Vname);
    end    

    N = size(V, 1)
    P = size(T, 1)
    
    savefolder_off = [sourcepath dataset '/off_2/'];
    savename_off = strcat(savefolder_off, sourcename, '.off');  % build off file (not really necessary, but needed to build matlab shape object)
    if ~exist(savefolder_off, 'dir')
       mkdir(savefolder_off);
    end
    
    % rotate shapes if needed so that they are y-aligned if this is not yet
    % the case (here setup for SHREC)
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    %if ismember(str2num(sourcename), [1,16,17,22,23,24,25,29,30,31,32,33,34,35,36,37,38,39])  % SHREC
    if ismember(str2num(sourcename), [16,17,22,23,24,25,29,30,31,32,33,34,35,36,37,38,39])  %SHREC_r
        ['ROTATING' sourcename]
        theta = (1/2) * pi;
        c = cos(theta);
        s = sin(theta);
        rotx = [1 0 0; 0 c -s; 0 s c];
        V = V * rotx;
    end
    
    if ismember(str2num(sourcename), [28])
        ['ROTATING' sourcename]
        theta = pi;
        c = cos(theta);
        s = sin(theta);
        rotx = [1 0 0; 0 c -s; 0 s c];
        V = V * rotx;
    end
    %%%%%%%%%%%
    %(here setup for SCAPE -- no need of setup for FAUST)
%     theta = (-1/2) * pi;
%     c = cos(theta);
%     s = sin(theta);
%     rotz = [c -s 0; s c 0; 0 0 1];
%     V = V * rotz;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    writeOFF(savename_off, V - mean(V, 1), T); % if we choose to center the shape (can be useful but then do not forget to center training shapes)
    max_spec_dim = 100;

    S1 = read_off_shape(savename_off);
    
    % BE CAREFUL to rescale with total surface parameter and save again if needed
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    S1 = MESH.compute_LaplacianBasis(S1, max_spec_dim);
    s = S1.sqrt_area;
    disp(['total area :' num2str(s)])
    S1.VERT = S1.surface.VERT/s;
    writeOFF(savename_off, S1.VERT, S1.surface.TRIV);
    S1 = read_off_shape(savename_off);
    %%%%%%%%%%%%%%%%%%%%%%%
    
    % compute laplacian data and save it in mat file
    
    S1 = MESH.compute_LaplacianBasis(S1, max_spec_dim);
    target_evals = S1.evals;
    target_evecs = S1.evecs;
    target_evecs_trans = S1.evecs' * S1.A;  % A could be saved instead, as it is less heavy. possible update

    %save data
    savefolder_spectral = [sourcepath dataset '/spectral/'];
    if ~exist(savefolder_spectral, 'dir')
       mkdir(savefolder_spectral)
    end
    save([savefolder_spectral sourcename '.mat'],'target_evals', 'target_evecs', 'target_evecs_trans');%,  '-v6');  %version can be added for compatibility
end

%% visualize spectral data, check if it is correct (not necessary)

% para.view_angle = [0, 180]; % adjust view angle
% para.rot_flag = true; % as there is a limit position on the coordinate system of Matlab, you can get the full rotation freedom by switching this parameter. 
% 
% figure;
% 
% for i = 1:max_spec_dim
%     plot_shape(S1, para); 
%     plot_function(S1, target_evecs(:, i), para);
%     title(num2str(i));
%     pause; 
% end

%% write dataset txt files (train, test, val)  (not always useful)

sourcefolder = ['../../../../media/donati/Data1/Datasets/' dataset];
filetype = 'off';
suffix = dataset(1:end - 1)

%trainptsfiles = dir(fullfile([sourcefolder 'points/'], 'val*.pts'));
%length(trainptsfiles)
files = dir(fullfile([sourcefolder filetype '/'], ['*.' filetype]));

%phase = 'training';
phase = 'test';
filesname = [sourcefolder suffix '_' phase '.txt'];

n_files = length(files)

fid = fopen(filesname,'wt');
for pts = 1:n_files
   pts;
   fprintf(fid, '%s\n', files(pts).name');
end
fclose(fid);
