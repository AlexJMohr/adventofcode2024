open Solutions.Day01
open Lib.Common

let () =
  Sys.chdir "../../../" |> ignore

  let test_part1 () =
    let input = read_input "inputs/tests/day01.txt" in
    let expected = 11 in
    Alcotest.(check int) "Part 1" expected (part1 input)

  let () = Printf.printf "Current working directory: %s\n" (Sys.getcwd ())

  let () = 
    let open Alcotest in
    run "Day 1 Tests" [
      ("Part 1", [test_case "Part 1" `Quick test_part1]);
    ]
