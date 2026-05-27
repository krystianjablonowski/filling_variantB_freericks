#!/usr/bin/env perl
use strict;
use warnings;
use Getopt::Long qw(GetOptions);
use Cwd qw(abs_path getcwd);
use File::Path qw(make_path);

my %opt = (
    "T-list" => "0.02",
    "w1-list" => "0.5,0.45,0.4,0.35,0.3",
    "U-min" => 0.0,
    "U-max" => 2.0,
    "n-U" => 21,
    "Delta-min" => 0.05,
    "Delta-max" => 2.0,
    "n-Delta" => 21,
    "Delta-chunk" => 1,
    "kind" => "typ",
    "omega0" => 12.0,
    "n-omega" => 501,
    "n-eps" => 51,
    "eta" => "1e-3",
    "max-iter" => 800,
    "tol" => "1e-4",
    "mix" => 0.12,
    "mu-min" => -8.0,
    "mu-max" => 8.0,
    "ne-tol" => "5e-3",
    "max-mu-iter" => 8,
    "mu-accept-diff" => "1e-3",
    "workers" => 1,
    "ppn" => 1,
    "walltime" => "12:00:00",
    "mem" => "12gb",
    "python" => "python3",
    "out" => "pbs_variantB_T002",
    "submit" => 0,
);

GetOptions(
    "T-list=s" => \$opt{"T-list"},
    "w1-list=s" => \$opt{"w1-list"},
    "U-min=f" => \$opt{"U-min"},
    "U-max=f" => \$opt{"U-max"},
    "n-U=i" => \$opt{"n-U"},
    "Delta-min=f" => \$opt{"Delta-min"},
    "Delta-max=f" => \$opt{"Delta-max"},
    "n-Delta=i" => \$opt{"n-Delta"},
    "Delta-chunk=i" => \$opt{"Delta-chunk"},
    "kind=s" => \$opt{"kind"},
    "omega0=f" => \$opt{"omega0"},
    "n-omega=i" => \$opt{"n-omega"},
    "n-eps=i" => \$opt{"n-eps"},
    "eta=s" => \$opt{"eta"},
    "max-iter=i" => \$opt{"max-iter"},
    "tol=s" => \$opt{"tol"},
    "mix=f" => \$opt{"mix"},
    "mu-min=f" => \$opt{"mu-min"},
    "mu-max=f" => \$opt{"mu-max"},
    "ne-tol=s" => \$opt{"ne-tol"},
    "max-mu-iter=i" => \$opt{"max-mu-iter"},
    "mu-accept-diff=s" => \$opt{"mu-accept-diff"},
    "workers=i" => \$opt{"workers"},
    "ppn=i" => \$opt{"ppn"},
    "walltime=s" => \$opt{"walltime"},
    "mem=s" => \$opt{"mem"},
    "python=s" => \$opt{"python"},
    "out=s" => \$opt{"out"},
    "queue=s" => \$opt{"queue"},
    "submit!" => \$opt{"submit"},
) or die "Bad options\n";

my $cwd = abs_path(getcwd());
my $pbs_dir = "$cwd/pbs_variantB_jobs";
my $out_dir = "$cwd/$opt{out}";
make_path($pbs_dir);
make_path($out_dir);

sub tag {
    my ($x) = @_;
    $x =~ s/\./p/g;
    $x =~ s/-/m/g;
    return $x;
}

sub linspace {
    my ($a, $b, $n) = @_;
    return ($a) if $n <= 1;
    my @x;
    for my $i (0 .. $n - 1) {
        push @x, $a + ($b - $a) * $i / ($n - 1);
    }
    return @x;
}

my @Ts = split /,/, $opt{"T-list"};
my @Deltas = linspace($opt{"Delta-min"}, $opt{"Delta-max"}, $opt{"n-Delta"});
my @pbs_files;

for my $T (@Ts) {
    for (my $start = 0; $start < @Deltas; $start += $opt{"Delta-chunk"}) {
        my $end = $start + $opt{"Delta-chunk"} - 1;
        $end = $#Deltas if $end > $#Deltas;
        my @chunk = @Deltas[$start .. $end];
        my $Delta_list = join(",", map { sprintf("%.16g", $_) } @chunk);
        my $name = "T_" . tag($T) . "__Didx_${start}_${end}__" . $opt{"kind"};
        my $chunk_out = "$out_dir/chunks/$name";
        make_path($chunk_out);
        my $pbs = "$pbs_dir/$name.pbs";
        open my $fh, ">", $pbs or die "Cannot write $pbs: $!";
        print $fh "#!/bin/bash\n";
        print $fh "#PBS -N vB_$name\n";
        print $fh "#PBS -l nodes=1:ppn=$opt{ppn}\n";
        print $fh "#PBS -l walltime=$opt{walltime}\n";
        print $fh "#PBS -l mem=$opt{mem}\n";
        print $fh "#PBS -o $chunk_out/job.out\n";
        print $fh "#PBS -e $chunk_out/job.err\n";
        print $fh "#PBS -q $opt{queue}\n" if defined $opt{queue};
        print $fh "cd '$cwd'\n";
        print $fh "$opt{python} solve_variantB_dmft.py ";
        print $fh "--T-list '$T' --w1-list '$opt{'w1-list'}' ";
        print $fh "--U-min $opt{'U-min'} --U-max $opt{'U-max'} --n-U $opt{'n-U'} ";
        print $fh "--Delta-list '$Delta_list' ";
        print $fh "--kind '$opt{kind}' ";
        print $fh "--omega0 $opt{omega0} --n-omega $opt{'n-omega'} --n-eps $opt{'n-eps'} ";
        print $fh "--eta $opt{eta} --max-iter $opt{'max-iter'} --tol $opt{tol} --mix $opt{mix} ";
        print $fh "--mu-min $opt{'mu-min'} --mu-max $opt{'mu-max'} --ne-tol $opt{'ne-tol'} --max-mu-iter $opt{'max-mu-iter'} ";
        print $fh "--mu-accept-diff $opt{'mu-accept-diff'} ";
        print $fh "--workers $opt{workers} --out '$chunk_out' --resume\n";
        close $fh;
        push @pbs_files, $pbs;
    }
}

my $merge = "$pbs_dir/merge_dmft_results.sh";
open my $mh, ">", $merge or die "Cannot write $merge: $!";
print $mh "#!/bin/bash\nset -e\n";
print $mh "cd '$cwd'\n";
print $mh "out='$out_dir/dmft_summary.csv'\n";
print $mh "first=1\n";
print $mh "rm -f \"\$out\"\n";
print $mh "for f in '$out_dir'/chunks/*/dmft_summary.csv; do\n";
print $mh "  [ -f \"\$f\" ] || continue\n";
print $mh "  if [ \$first -eq 1 ]; then cat \"\$f\" > \"\$out\"; first=0; else tail -n +2 \"\$f\" >> \"\$out\"; fi\n";
print $mh "done\n";
print $mh "echo \"Merged DMFT CSV: \$out\"\n";
close $mh;
chmod 0755, $merge;

print "Prepared " . scalar(@pbs_files) . " PBS jobs\n";
print "Project dir: $cwd\n";
print "Output dir:  $out_dir\n";
print "PBS scripts written to: $pbs_dir\n";
print "Merge script: $merge\n";
print "workers/job=$opt{workers}, ppn=$opt{ppn}\n";

if ($opt{submit}) {
    for my $pbs (@pbs_files) {
        system("qsub", $pbs) == 0 or warn "qsub failed for $pbs\n";
    }
} else {
    print "\nNot submitted. Add --submit to run jobs.\n";
    print "After all jobs finish:\n";
    print "  bash '$merge'\n";
}
