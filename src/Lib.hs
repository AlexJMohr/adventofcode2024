module Lib (readFileFromArgs, distance) where

import System.Environment (getArgs)
import System.Exit (die)

readFileFromArgs :: IO String
readFileFromArgs = do
  args <- getArgs
  case args of
    (file : _) -> readFile file
    _ -> die "Usage: day01 <filename>"

distance :: Num a => a -> a -> a
distance = (abs .) . (-)