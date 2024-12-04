let read_input filepath =
  let ic = open_in filepath in
  let buf = Buffer.create 1024 in
  try
    while true do
      Buffer.add_string buf (input_line ic);
      Buffer.add_char buf '\n'
    done;
    Buffer.contents buf
  with End_of_file ->
    close_in ic;
    Buffer.contents buf