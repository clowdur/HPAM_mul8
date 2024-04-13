// Oliver Huang & Jared Yoder
// EE 478 Capstone Project
// testbench for multipliers
//`default_nettype None

//`timescale 1ns/1ps

module testbench;

  //logic clk;
  logic [15:0] result; 
  logic [7:0] a, b;
  real true_result;
  real rel_error;
  
  assign true_result = (a && b) ? (real'(a)*real'(b)) : real'(0.000000000000001); //approximate zero to avoid divide by zero
  assign rel_error = (a && b || (result != 0)) ? ((real'(result)-true_result)/true_result) : real'(0);
  
  //parameter T = 20;
  // define simulated clock if sequential logic is used
  // initial begin
  //   clk <= 0;
  //   forever #(T/2) clk <= ~clk;
  // end // initial clock

  wallaceTreeMultiplier8Bit dut (.*);
  //product exact_product (.res(true_result), .A(a), .B(b));
  

  initial begin
    $write("%c[1;32m",27); $display("\nBegin: Multiplier Test Cases\n"); #20;
    $write("%c[0m",27);
    a = 255; b = 255; #20;
    $display("inputs: a:%d, b:%d", a, b); #20;
    $display("result: %d  true: %.1f  error: %f\n", result, true_result, rel_error); #20;
    a = 17; b = 17; #20;
    $display("inputs: a:%d, b:%d", a, b); #20;
    $display("result: %d  true: %.1f  error: %f\n", result, true_result, rel_error); #20;
    a = 23; b = 67; #20;
    $display("inputs: a:%d, b:%d", a, b); #20;
    $display("result: %d  true: %.1f  error: %f\n", result, true_result, rel_error); #20;
    a = 67; b = 23; #20;
    $display("inputs: a:%d, b:%d", a, b); #20;
    $display("result: %d  true: %.1f  error: %f\n", result, true_result, rel_error); #20;
    a = 0; b = 19; #20;
    $display("inputs: a:%d, b:%d", a, b); #20;
    $display("result: %d  true: %.1f  error: %f\n", result, true_result, rel_error); #20;
    a = 19; b = 0; #20;
    $display("inputs: a:%d, b:%d", a, b); #20;
    $display("result: %d  true: %.1f  error: %f\n", result, true_result, rel_error); #20;
    a = 0; b = 0; #20;
    $display("inputs: a:%d, b:%d", a, b); #20;
    $display("result: %d  true: %.1f  error: %f\n", result, true_result, rel_error); #20;
    $write("%c[1;32m",27); $display("End: Multiplier Test Cases\n"); #20;
    $write("%c[0m",27);
  end

endmodule