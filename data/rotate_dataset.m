% Martin Kersner, m.kersner@gmail.com
% 2015/11/07

% Rotate images and enlarge dataset.

function rotate_dataset(angle, csv_file, new_csv_file, dataset_path_input, dataset_path_output)
    ext = '.png';

    % load
    fid = fopen(csv_file);
    new_fid = fopen(new_csv_file, 'w');
    out = textscan(fid, '%s%s', 'delimiter', ',');
    
    img_names = out{1};
    labels    = out{2};

    new_img_names = {};
    new_labels = {};
    
    total = size(img_names, 1);
    
    % rotate
    for i = 1:total
        full_img_name = fullfile(dataset_path_input, [img_names{i}, '.png']);
        img = imread(full_img_name);
    
        rotate_left  = rotate_image(img, angle);
        rotate_right = rotate_image(img, -angle);
        
        % save
        left_img_name  = [img_names{i} '-' labels{i} '-left'];
        orig_img_name  = [img_names{i} '-' labels{i} '-orig'];
        right_img_name = [img_names{i} '-' labels{i} '-right'];

        left_img_path  = fullfile(dataset_path_output, [left_img_name  ext]);
        orig_img_path  = fullfile(dataset_path_output, [orig_img_name  ext]);
        right_img_path = fullfile(dataset_path_output, [right_img_name ext]);

        imwrite(rotate_left, left_img_path);
        imwrite(img, orig_img_path);
        imwrite(rotate_right, right_img_path);

        % write to output csv file
        fprintf(new_fid,'%s, %s\n', left_img_name,  labels{i});
        fprintf(new_fid,'%s, %s\n', orig_img_name,  labels{i});
        fprintf(new_fid,'%s, %s\n', right_img_name, labels{i});
    end

    fclose(fid);
    fclose(new_fid);
end

function final_rotated_img = rotate_image(img, angle)
    img = gray2rgb(img);
    rotated_img = imrotate(img, angle, 'crop');
    
    fg = uint8(rgb2gray(rotated_img) > 0);
    bg = 1 - fg;
    
    final_rotated_img = img.*repmat(bg, [1,1,3]) + rotated_img.*repmat(fg, [1,1,3]);
end

function rgb = gray2rgb(gray)
    if (length(size(gray)) == 2)
        rgb = repmat(gray, [1,1,3]);
    else
        rgb = gray;
    end
end
