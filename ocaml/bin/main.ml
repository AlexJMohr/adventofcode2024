open Lib.Common
open Solutions

let () =
  (* print cwd *)
  Printf.printf "Current working directory: %s\n" (Sys.getcwd ());

  if Array.length Sys.argv < 4 then
    Printf.printf "Usage: advent_of_code <day> <part> <input_file>\n"
  else
    let day = int_of_string Sys.argv.(1) in
    let part = int_of_string Sys.argv.(2) in
    let input = read_input Sys.argv.(3) in
    match day with
    | 1 -> Day01.run part input
    | _ -> Printf.printf "Day %d not implemented.\n" day