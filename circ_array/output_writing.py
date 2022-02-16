import os
from array_info import array
import pandas as pd
import numpy as np
from slow_vec_calcs import calculate_locus
from geo_sphere_calcs import predict_pierce_points

def write_to_file_check_lines(filepath, header, newlines, strings):
    """
    Reads lines in a file and replaces those which match all the
    strings in "strings" with the newlines.

    Parameters
    ----------
    filepath : string
        Name and path of results file.

    header : string
        Header to write to file if it does not exist.

    newlines : list of strings
        newlines to be added to file.

    strings : list of strings
        Strings to identify lines which need to be replaced in the
        file at filepath.

    Returns
    -------
    Nothing

    """

    found = False
    added = (
        False  # just so i dont write it twice if i find the criteria in multiple lines
    )
    ## write headers to the file if it doesnt exist
    line_list = []

    if os.path.exists(filepath):
        with open(filepath, "r") as Multi_file:
            for line in Multi_file:
                if all(x in line for x in strings):
                    print("strings in line, replacing")
                    # to avoid adding the new lines multiple times
                    if added == False:
                        line_list.extend(newlines)
                        added = True
                    else:
                        print("already added to file")
                    found = True
                else:
                    line_list.append(line)
    else:
        with open(filepath, "w") as Multi_file:
            Multi_file.write(header)
            line_list.append(header)
    if not found:
        print("strings not in file. Adding to the end.")
        line_list.extend(newlines)
    else:
        pass

    with open(filepath, "w") as Multi_file2:
        Multi_file2.write("".join(line_list))


    return

def write_to_file(filepath, st, peaks, prediction, phase, time_window):
    """
    Function to write event and station information with slowness vector
    properties to a results file.

    Parameters
    ----------
    filepath : string
        Name and path of results file.

    st : Obspy stream object
        Stream object of SAC files assumed to have headers populated
        as described in the README.

    peaks : 2D numpy array of floats
        2D array of floats [[baz, slow]]
        for the arrival locations.

    prediction : 2D numpy array of floats
        2D numpy array of floats of the predicted arrival
        in [[baz, slow]].

    phase : string
        Target phase (e.g. SKS)

    time_window : 1D numpy array of floats
        numpy array of floats describing the start and end
        of time window in seconds.

    Returns
    -------
        Nothing.
    """
    from array_info import array
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    else:
        pass

    array = array(st)
    event_time = array.eventtime()
    geometry = array.geometry()
    distances = array.distances(type="deg")
    mean_dist = np.mean(distances)
    stations = array.stations()
    centre_x, centre_y = np.mean(geometry[:, 0]), np.mean(geometry[:, 1])
    sampling_rate = st[0].stats.sampling_rate
    evdp = st[0].stats.sac.evdp
    evla = st[0].stats.sac.evla
    evlo = st[0].stats.sac.evlo

    name = (
        str(event_time.year)
        + f"{event_time.month:02d}"
        + f"{event_time.day:02d}"
        + "_"
        + f"{event_time.hour:02d}"
        + f"{event_time.minute:02d}"
        + f"{event_time.second:02d}"
    )
    stat_string = ",".join(stations)
    newlines = []

    # make the line string
    for peak in peaks:
        baz_obs = peak[0]
        baz_pred = prediction[0]
        baz_diff = baz_obs - baz_pred

        slow_obs = peak[1]
        slow_pred = prediction[1]
        slow_diff = slow_obs - slow_pred

        newline = (
            name
            + f" {evlo:.2f} {evla:.2f} {evdp:.2f} {centre_x:.2f} {centre_y:.2f} {baz_pred:.2f} {baz_obs:.2f} {baz_diff:.2f} {slow_pred:.2f} {slow_obs:.2f} {slow_diff:.2f} "
            + stat_string
            + f" {time_window[0]:.2f} {time_window[1]:.2f} "
            + phase
            + " \n"
        )
        # %(name, evlo, evla, evdp, centre_x, centre_y,  baz_pred, baz_obs, baz_diff, slow_pred, slow_obs, slow_diff, ','.join(stations), time_window[0], time_window[1], phase)
        # there will be multiple lines so add these to this list.
        newlines.append(newline)

    header = "name evlo evla evdp stlo_mean stla_mean pred_baz baz_obs baz_diff pred_slow slow_obs slow_diff stations start_window end_window phase \n"

    write_to_file_check_lines(filepath, header, newlines, strings = [name, phase, f"{centre_y:.2f}"])

    return


def create_plotting_file(filepath,
                         outfile,
                         depth,
                         locus_file="./Locus_results.txt",
                         locus=False,
                         mod='prem'):
    """
    Extract the plotting information from the results file and calculate
    pierce points at depth.

    Parameters
    ----------
    filepath : string
        Path to results file.

    outfile : string
        Path to plotting file.

    depth : float
        Depth to calculate pierce points for.

    locus_file : string
        Path to file for locus calculation result default is
        "./Locus_results.txt".

    locus : bool
        Do you want to calculate the locus, default is False.

    mod : string
        1D velocity model to use for predicting pierce point locations.
        Default is prem.

    Returns
    -------
    Nothing
    """


    results_df = pd.read_csv(filepath, sep=' ', index_col=False)

    newlines = []
    locus_newlines = []
    header = "Name evla evlo evdp stla_mean stlo_mean slow_pred slow_diff slow_std_dev baz_pred baz_diff baz_std_dev del_x_slow del_y_slow mag az error_ellipse_area multi phase s_pierce_la s_pierce_lo r_pierce_la r_pierce_lo s_reloc_pierce_la s_reloc_pierce_lo r_reloc_pierce_la r_reloc_pierce_lo\n"
    locus_header = 'Name evla evlo evdp stla stlo s_pierce_la s_pierce_lo r_pierce_la r_pierce_lo s_reloc_pierce_la s_reloc_pierce_lo r_reloc_pierce_la r_reloc_pierce_lo Phi_1 Phi_2'


    newlines.append(header)
    locus_newlines.append(locus_header)
    for index, row in results_df.iterrows():
        print(row)
        dir = row['dir']
        name = row['Name']
        evla = row['evla']
        evlo = row['evlo']
        evdp = row['evdp']
        reloc_evla=row['reloc_evla']
        reloc_evlo=row['reloc_evlo']
        stla_mean = row['stla_mean']
        stlo_mean = row['stlo_mean']
        baz = row['baz_max']
        slow = row['slow_max']
        baz_pred = row['baz_pred']
        slow_pred = row['slow_pred']
        mag = row['mag']
        az = row['az']
        phase = row['phase']
        multi = row['multi']


        s_pierce_la, s_pierce_lo, r_pierce_la, r_pierce_lo = predict_pierce_points(evla=evla,
                                                                                   evlo=evlo,
                                                                                   evdp=evdp,
                                                                                   stla=stla_mean,
                                                                                   stlo=stlo_mean,
                                                                                   phase=phase,
                                                                                   target_depth=depth,
                                                                                   mod=mod)


        # relocate event to match baz and slow
        # reloc_evla, reloc_evlo = relocate_event_baz_slow(evla=evla,
        #                                                 evlo=evlo,
        #                                                 evdp=evdp,
        #                                                 stla=stla_mean,
        #                                                 stlo=stlo_mean,
        #                                                 baz=baz,
        #                                                 slow=slow,
        #                                                 phase=phase,
        #                                                 mod=mod)


        try:
            s_reloc_pierce_la, s_reloc_pierce_lo, r_reloc_pierce_la, r_reloc_pierce_lo = predict_pierce_points(evla=reloc_evla,
                                                                                                               evlo=reloc_evlo,
                                                                                                               evdp=evdp,
                                                                                                               stla=stla_mean,
                                                                                                               stlo=stlo_mean,
                                                                                                               phase=phase,
                                                                                                               target_depth=depth,
                                                                                                               mod=mod)

            newline = list(row[["Name", "evla", "evlo", "evdp", "stla_mean", "stlo_mean", "slow_pred", "slow_diff", "slow_std_dev",
                           "baz_pred", "baz_diff", "baz_std_dev", "del_x_slow", "del_y_slow", "mag", "az", "error_ellipse_area", "multi",
                           "phase"]].astype(str))


            for new_item in [s_pierce_la, s_pierce_lo, r_pierce_la, r_pierce_lo, s_reloc_pierce_la, s_reloc_pierce_lo, r_reloc_pierce_la, r_reloc_pierce_lo]:
                newline.append(new_item)

            newlines.append(' '.join(newline) + "\n")

        except:
            # print(f"{phase} doesnt have a predicted arrival, using ScS instead")
            s_reloc_pierce_la, s_reloc_pierce_lo, r_reloc_pierce_la, r_reloc_pierce_lo = predict_pierce_points(evla=reloc_evla,
                                                                                                               evlo=reloc_evlo,
                                                                                                               evdp=evdp,
                                                                                                               stla=stla_mean,
                                                                                                               stlo=stlo_mean,
                                                                                                               phase="ScS",
                                                                                                               target_depth=depth,
                                                                                                               mod=mod)

            newline = list(row[["Name", "evla", "evlo", "evdp", "stla_mean", "stlo_mean", "slow_pred", "slow_diff", "slow_std_dev",
                           "baz_pred", "baz_diff", "baz_std_dev", "del_x_slow", "del_y_slow", "mag", "az", "error_ellipse_area", "multi"]].astype(str))


            for new_item in ['ScS', s_pierce_la, s_pierce_lo, r_pierce_la, r_pierce_lo, s_reloc_pierce_la, s_reloc_pierce_lo, r_reloc_pierce_la, r_reloc_pierce_lo]:
                newline.append(new_item)

            newlines.append(' '.join(newline) + "\n")

        if locus == True:
            if dir and name not in locus_newlines:
                if multi == 'y':
                    # get the points of the multipathed arrivals
                    multi_df = results_df.loc[results_df['dir'] == dir]
                    number_arrivals = len(multi_df)

                    # get smallest baz value
                    min_baz = multi_df.min()['baz_diff']

                    p1_df = multi_df.loc[multi_df['baz_diff'] == min_baz]
                    P1 = [p1_df['slow_x_obs'].values[0], p1_df['slow_y_obs'].values[0]]
                    for i,row in multi_df.iterrows():

                        if row['baz_diff'] != min_baz:
                            P2 = [float(row['slow_x_obs']), float(row['slow_y_obs'])]
                            Theta, Midpoint, Phi_1, Phi_2 = calculate_locus(P1, P2)
                            linelist = list(np.array([name, evla, evlo, stla_mean, stlo_mean, s_pierce_la, s_pierce_lo, r_pierce_la, r_pierce_lo, s_reloc_pierce_la, s_reloc_pierce_lo, r_reloc_pierce_la, r_reloc_pierce_lo, Phi_1, Phi_2]).astype(str))
                            locus_newlines.append(' '.join(linelist))

    # ok! now write these newlines to a file of the users choosing
    # note for this we are basically removing everything this sub array found
    # and replacing it with the new lines.

    # overwrite the previous plotting file
    with open(outfile, 'w') as rewrite:
        pass

    with open(outfile, 'a') as outfile:
        for outline in newlines:
            outfile.write(outline)

    if locus == True:

        with open(locus_file, 'w') as rewrite:
            pass

        with open(locus_file, 'a') as outfile:
            for outline in locus_newlines:
                outfile.write(outline)


    return
